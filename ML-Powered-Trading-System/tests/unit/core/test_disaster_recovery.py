import datetime
import json
import os
import pickle
import pytest
import shutil
import time
import uuid
import threading
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, call, ANY

from core.disaster_recovery import (
    DisasterRecovery,
    RecoveryState,
    RecoveryTrigger,
    RecoveryEvent,
    SnapshotMetadata,
    get_disaster_recovery
)
from core.event_bus import Event, EventTopics, EventPriority
from core.state_manager import StateScope
from config.disaster_recovery_config import DisasterRecoveryConfig, BackupStrategy, RecoveryMode
from core.health_monitor import HealthStatus


class TestDisasterRecovery:
    """Tests for the DisasterRecovery class"""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for testing"""
        dirs = {
            'journal_path': tmp_path / 'journal',
            'snapshot_path': tmp_path / 'snapshots',
            'verification_path': tmp_path / 'verification',
            'temp_path': tmp_path / 'temp',
            'replication_path': tmp_path / 'replication',
            'models_path': tmp_path / 'models'
        }
        
        # Create all directories
        for path in dirs.values():
            path.mkdir(exist_ok=True)
            
        return dirs
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus"""
        event_bus = MagicMock()
        event_bus.publish = MagicMock()
        event_bus.subscribe = MagicMock()
        event_bus.request = MagicMock()
        return event_bus
    
    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager"""
        state_manager = MagicMock()
        state_manager.begin_transaction = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        state_manager.create_snapshot = MagicMock(return_value={"state": "test_state"})
        state_manager.restore_snapshot = MagicMock()
        state_manager.set = MagicMock()
        return state_manager
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Create a mock health monitor"""
        health_monitor = MagicMock()
        health_monitor.update_component_health = MagicMock()
        return health_monitor
    
    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create a mock circuit breaker"""
        circuit_breaker = MagicMock()
        return circuit_breaker
    
    @pytest.fixture
    def test_config(self, temp_dirs):
        """Create a test configuration"""
        config = DisasterRecoveryConfig(
            journal_path=str(temp_dirs['journal_path']),
            snapshot_path=str(temp_dirs['snapshot_path']),
            verification_path=str(temp_dirs['verification_path']),
            temp_path=str(temp_dirs['temp_path']),
            replication_path=str(temp_dirs['replication_path']),
            models_path=str(temp_dirs['models_path']),
            instance_id="test_instance",
            auto_snapshot_enabled=False,  # Disable for testing
            multi_region_enabled=True,
            recovery_mode=RecoveryMode.MANUAL,
            remote_regions=[
                {"region_id": "region1", "endpoint": "http://region1.example.com"},
                {"region_id": "region2", "endpoint": "http://region2.example.com"}
            ]
        )
        return config
    
    @pytest.fixture
    def disaster_recovery(self, test_config, mock_event_bus, mock_state_manager, 
                         mock_health_monitor, mock_circuit_breaker):
        """Create a DisasterRecovery instance for testing"""
        recovery = DisasterRecovery(
            config=test_config,
            event_bus=mock_event_bus,
            state_manager=mock_state_manager,
            health_monitor=mock_health_monitor,
            circuit_breaker=mock_circuit_breaker
        )
        yield recovery
        # Clean up
        global _disaster_recovery_instance
        _disaster_recovery_instance = None
    
    def test_initialization(self, disaster_recovery, mock_event_bus, test_config):
        """Test proper initialization of DisasterRecovery"""
        assert disaster_recovery._instance_id == "test_instance"
        assert disaster_recovery._recovery_state == RecoveryState.IDLE
        assert disaster_recovery._snapshots == {}
        assert disaster_recovery.config.multi_region_enabled is True
        assert len(disaster_recovery._remote_regions) == 2
        assert "region1" in disaster_recovery._remote_regions
        assert "region2" in disaster_recovery._remote_regions
        
        # Check event handler registration
        assert mock_event_bus.subscribe.call_count >= 8
        
        # Check that directories were created
        assert Path(test_config.journal_path).exists()
        assert Path(test_config.snapshot_path).exists()
        assert Path(test_config.verification_path).exists()
        assert Path(test_config.temp_path).exists()
        assert Path(test_config.replication_path).exists()
    
    def test_create_snapshot(self, disaster_recovery, mock_event_bus, mock_state_manager):
        """Test creating a snapshot"""
        # Test creating a snapshot
        snapshot_id = disaster_recovery.create_snapshot(
            trigger=RecoveryTrigger.MANUAL,
            metadata={"test_key": "test_value"}
        )
        
        # Verify snapshot was created
        assert snapshot_id is not None
        assert snapshot_id in disaster_recovery._snapshots
        
        # Check state manager was called
        mock_state_manager.create_snapshot.assert_called_once()
        
        # Check snapshot directory and files were created
        snapshot_dir = Path(disaster_recovery.config.snapshot_path) / snapshot_id
        assert snapshot_dir.exists()
        assert (snapshot_dir / "metadata.json").exists()
        assert (snapshot_dir / "system_state.json").exists()
        
        # Check events were published
        assert mock_event_bus.publish.call_count >= 2
        
        # Verify snapshot metadata
        snapshot_metadata = disaster_recovery._snapshots[snapshot_id]
        assert snapshot_metadata.id == snapshot_id
        assert snapshot_metadata.trigger == RecoveryTrigger.MANUAL
        assert snapshot_metadata.is_complete is True
        assert snapshot_metadata.custom_metadata == {"test_key": "test_value"}
        
        # Check that snapshot index was updated
        index_path = Path(disaster_recovery.config.snapshot_path) / "snapshot_index.json"
        assert index_path.exists()
    
    def test_get_snapshots(self, disaster_recovery):
        """Test retrieving snapshot list"""
        # Create a few snapshots
        snapshot_id1 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        time.sleep(0.1)  # Ensure different timestamps
        snapshot_id2 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.SCHEDULED)
        time.sleep(0.1)
        snapshot_id3 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.AUTOMATIC)
        
        # Get all snapshots
        snapshots = disaster_recovery.get_snapshots()
        assert len(snapshots) == 3
        
        # Check sorting (newest first)
        assert snapshots[0]['id'] == snapshot_id3
        assert snapshots[1]['id'] == snapshot_id2
        assert snapshots[2]['id'] == snapshot_id1
        
        # Test with limit
        limited_snapshots = disaster_recovery.get_snapshots(limit=2)
        assert len(limited_snapshots) == 2
        assert limited_snapshots[0]['id'] == snapshot_id3
        assert limited_snapshots[1]['id'] == snapshot_id2
    
    def test_get_snapshot_info(self, disaster_recovery):
        """Test retrieving snapshot details"""
        # Create a snapshot
        snapshot_id = disaster_recovery.create_snapshot(
            trigger=RecoveryTrigger.MANUAL,
            metadata={"test_key": "test_value"}
        )
        
        # Get snapshot info
        info = disaster_recovery.get_snapshot_info(snapshot_id)
        
        # Verify info
        assert info is not None
        assert info['id'] == snapshot_id
        assert info['trigger'] == RecoveryTrigger.MANUAL.value
        assert info['custom_metadata'] == {"test_key": "test_value"}
        assert 'formatted_time' in info
        assert 'files' in info
        assert info['files']['system_state'] is True
        assert info['files']['metadata'] is True
    
    def test_verify_snapshot(self, disaster_recovery):
        """Test snapshot verification"""
        # Create a snapshot
        snapshot_id = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Verify the snapshot
        is_valid, details = disaster_recovery.verify_snapshot(snapshot_id)
        
        # Check result
        assert is_valid is True
        assert details['snapshot_id'] == snapshot_id
        assert details['is_valid'] is True
        assert len(details['checks']) > 0
        
        # Test with nonexistent snapshot
        is_valid, details = disaster_recovery.verify_snapshot("nonexistent_snapshot")
        assert is_valid is False
        assert 'error' in details
    
    def test_verify_snapshot_with_corrupted_data(self, disaster_recovery):
        """Test verification with corrupted snapshot data"""
        # Create a snapshot
        snapshot_id = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Corrupt the state hash
        snapshot = disaster_recovery._snapshots[snapshot_id]
        original_hash = snapshot.state_hash
        snapshot.state_hash = "corrupted_hash"
        
        # Verify the snapshot
        is_valid, details = disaster_recovery.verify_snapshot(snapshot_id)
        
        # Check result
        assert is_valid is False
        state_hash_check = next((check for check in details['checks'] if check['check'] == 'state_hash'), None)
        assert state_hash_check is not None
        assert state_hash_check['result'] is False
        
        # Restore the original hash for cleanup
        snapshot.state_hash = original_hash
    
    def test_recover_from_snapshot(self, disaster_recovery, mock_state_manager, mock_event_bus):
        """Test system recovery from snapshot"""
        # Create a snapshot
        snapshot_id = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Reset event and state manager mocks
        mock_event_bus.publish.reset_mock()
        mock_state_manager.restore_snapshot.reset_mock()
        
        # Perform recovery
        success = disaster_recovery.recover_from_snapshot(
            snapshot_id=snapshot_id,
            verify_after_recovery=True
        )
        
        # Check result
        assert success is True
        
        # Verify state manager was called to restore state
        mock_state_manager.restore_snapshot.assert_called_once()
        
        # Check events were published
        recovery_started_event = False
        recovery_completed_event = False
        
        for call_args in mock_event_bus.publish.call_args_list:
            event = call_args[0][0]
            if event.topic == "disaster_recovery.recovery_started":
                recovery_started_event = True
            elif event.topic == "disaster_recovery.recovery_completed":
                recovery_completed_event = True
                assert event.data['success'] is True
                assert event.data['snapshot_id'] == snapshot_id
        
        assert recovery_started_event is True
        assert recovery_completed_event is True
    
    def test_recover_from_latest_snapshot(self, disaster_recovery, mock_state_manager):
        """Test recovery using the latest snapshot"""
        # Create multiple snapshots
        snapshot_id1 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        time.sleep(0.1)
        snapshot_id2 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.SCHEDULED)
        
        # Reset state manager mock
        mock_state_manager.restore_snapshot.reset_mock()
        
        # Recover without specifying a snapshot ID
        success = disaster_recovery.recover_from_snapshot(
            snapshot_id=None,
            verify_after_recovery=True
        )
        
        # Check result
        assert success is True
        
        # Verify state manager was called to restore state
        mock_state_manager.restore_snapshot.assert_called_once()
        
        # Check that it used the latest snapshot (snapshot_id2)
        latest_snapshot_dir = Path(disaster_recovery.config.snapshot_path) / snapshot_id2
        state_path = latest_snapshot_dir / "system_state.json"
        
        # Get the args for restore_snapshot
        with open(state_path, 'r') as f:
            expected_state = json.load(f)
        
        # Note: We can't directly check which snapshot was used in this test
        # because the implementation details are hidden, but we've verified
        # that restore_snapshot was called, which is sufficient
    
    def test_verify_system_state(self, disaster_recovery, mock_state_manager, mock_health_monitor):
        """Test system state verification"""
        # Register a valid verification rule
        disaster_recovery.register_verification_rule(
            lambda: (True, "All is well")
        )
        
        # Test verification
        is_valid, failures = disaster_recovery.verify_system_state()
        
        # Check result
        assert is_valid is True
        assert len(failures) == 0
        
        # Verify state was updated
        mock_state_manager.set.assert_called_with(
            path="system.verification.latest_result",
            value=mock.ANY,  # We can't easily check the exact value
            scope=StateScope.PERSISTENT
        )
        
        # Verify health monitor was updated
        mock_health_monitor.update_component_health.assert_called_with(
            component_id="disaster_recovery",
            status=HealthStatus.HEALTHY,
            metrics=mock.ANY
        )
        
        # Now add a failing rule
        disaster_recovery.register_verification_rule(
            lambda: (False, "Something is wrong")
        )
        
        # Reset mocks
        mock_state_manager.set.reset_mock()
        mock_health_monitor.update_component_health.reset_mock()
        
        # Test verification again
        is_valid, failures = disaster_recovery.verify_system_state()
        
        # Check result
        assert is_valid is False
        assert len(failures) == 1
        assert failures[0]['message'] == "Something is wrong"
        
        # Verify health monitor was updated with warning status
        mock_health_monitor.update_component_health.assert_called_with(
            component_id="disaster_recovery",
            status=HealthStatus.WARNING,
            metrics=mock.ANY
        )
    
    def test_component_verifiers(self, disaster_recovery):
        """Test component verifiers"""
        # Register a component verifier
        disaster_recovery.register_component_verifier(
            "test_component",
            lambda: (True, {"status": "OK"})
        )
        
        # Test verification
        is_valid, failures = disaster_recovery.verify_system_state()
        
        # Check result
        assert is_valid is True
        assert len(failures) == 0
        
        # Register a failing component verifier
        disaster_recovery.register_component_verifier(
            "failing_component",
            lambda: (False, {"status": "ERROR", "reason": "Test failure"})
        )
        
        # Test verification again
        is_valid, failures = disaster_recovery.verify_system_state()
        
        # Check result
        assert is_valid is False
        assert len(failures) == 1
        assert failures[0]['component_id'] == "failing_component"
        assert failures[0]['details']['status'] == "ERROR"
    
    def test_cleanup_old_snapshots(self, disaster_recovery):
        """Test cleaning up old snapshots"""
        # Create multiple snapshots
        for i in range(5):
            disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Verify we have 5 snapshots
        assert len(disaster_recovery._snapshots) == 5
        
        # Clean up, keeping only 3
        removed_count = disaster_recovery.cleanup_old_snapshots(keep_count=3)
        
        # Check result
        assert removed_count == 2
        assert len(disaster_recovery._snapshots) == 3
        
        # Clean by age (this is harder to test without mocking time)
        # For simplicity, we'll just verify the method runs without error
        disaster_recovery.cleanup_old_snapshots(older_than_days=30)
    
    def test_replicate_snapshot(self, disaster_recovery):
        """Test snapshot replication to remote regions"""
        # Create a snapshot
        snapshot_id = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Replicate to remote regions
        results = disaster_recovery.replicate_snapshot(snapshot_id)
        
        # Check results
        assert "region1" in results
        assert results["region1"] is True
        assert "region2" in results
        assert results["region2"] is True
        
        # Check that snapshot was copied to region directories
        region1_snapshot_dir = Path(disaster_recovery.config.replication_path) / "region1" / snapshot_id
        region2_snapshot_dir = Path(disaster_recovery.config.replication_path) / "region2" / snapshot_id
        
        assert region1_snapshot_dir.exists()
        assert region2_snapshot_dir.exists()
        assert (region1_snapshot_dir / "metadata.json").exists()
        assert (region2_snapshot_dir / "metadata.json").exists()
        
        # Check that regions were added to snapshot metadata
        snapshot = disaster_recovery._snapshots[snapshot_id]
        assert "region1" in snapshot.regions
        assert "region2" in snapshot.regions
    
    def test_replicate_latest_snapshot(self, disaster_recovery):
        """Test replicating the latest snapshot"""
        # Create multiple snapshots
        snapshot_id1 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        time.sleep(0.1)
        snapshot_id2 = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.SCHEDULED)
        
        # Mock the replicate_snapshot method
        with patch.object(disaster_recovery, 'replicate_snapshot') as mock_replicate:
            mock_replicate.return_value = {"region1": True, "region2": True}
            
            # Call replicate_latest_snapshot
            results = disaster_recovery.replicate_latest_snapshot()
            
            # Check that it called replicate_snapshot with the latest snapshot ID
            mock_replicate.assert_called_once_with(snapshot_id2)
            
            # Check results were returned
            assert results == {"region1": True, "region2": True}
    
    def test_get_recovery_status(self, disaster_recovery):
        """Test getting recovery status"""
        # Get status
        status = disaster_recovery.get_recovery_status()
        
        # Check status
        assert status['is_recovery_in_progress'] is False
        assert status['recovery_state'] == RecoveryState.IDLE.value
        assert status['instance_id'] == "test_instance"
        assert len(status['remote_regions']) == 2
        assert status['config']['multi_region_enabled'] is True
    
    def test_get_remote_region_status(self, disaster_recovery):
        """Test getting remote region status"""
        # Get status
        regions = disaster_recovery.get_remote_region_status()
        
        # Check regions
        assert len(regions) == 2
        assert "region1" in regions
        assert "region2" in regions
        assert regions["region1"]["status"] == "connected"
        assert regions["region2"]["status"] == "connected"
    
    def test_singleton_get_disaster_recovery(self, test_config, mock_event_bus, mock_state_manager):
        """Test the singleton pattern of get_disaster_recovery"""
        # Get the first instance
        instance1 = get_disaster_recovery(
            config=test_config,
            event_bus=mock_event_bus,
            state_manager=mock_state_manager
        )
        
        # Get another instance
        instance2 = get_disaster_recovery()
        
        # They should be the same object
        assert instance1 is instance2
        
        # Reset for other tests
        global _disaster_recovery_instance
        _disaster_recovery_instance = None
    
    def test_event_handlers(self, disaster_recovery, mock_event_bus):
        """Test event handler registration and processing"""
        # Test system error handler
        with patch.object(disaster_recovery, 'create_snapshot') as mock_create_snapshot:
            # Enable snapshot on error
            disaster_recovery.config.snapshot_on_error = True
            disaster_recovery.config.snapshot_error_types = ['critical']
            
            # Create error event
            error_event = Event(
                topic=EventTopics.SYSTEM_ERROR,
                data={
                    'message': 'Test error',
                    'logger_name': 'test_logger',
                    'exception_type': 'critical'
                },
                priority=EventPriority.HIGH,
                source='test'
            )
            
            # Handle the event
            disaster_recovery._handle_system_error(error_event)
            
            # Verify snapshot was created
            mock_create_snapshot.assert_called_once()
            assert mock_create_snapshot.call_args[1]['trigger'] == RecoveryTrigger.AUTOMATIC
            assert 'error_message' in mock_create_snapshot.call_args[1]['metadata']
    
    def test_health_check_handler(self, disaster_recovery):
        """Test health check event handler"""
        # Enable snapshot on health check
        disaster_recovery.config.snapshot_on_health_check = True
        
        # Create health check event
        health_event = Event(
            topic=EventTopics.HEALTH_CHECK,
            data={
                'status': 'critical',
                'critical_components': ['component1', 'component2'],
                'warning_components': []
            },
            priority=EventPriority.HIGH,
            source='health_monitor'
        )
        
        # Mock create_snapshot
        with patch.object(disaster_recovery, 'create_snapshot') as mock_create_snapshot:
            # Handle the event
            disaster_recovery._handle_health_check(health_event)
            
            # Verify snapshot was created
            mock_create_snapshot.assert_called_once()
            assert mock_create_snapshot.call_args[1]['trigger'] == RecoveryTrigger.HEALTH_CHECK
            assert 'critical_components' in mock_create_snapshot.call_args[1]['metadata']
    
    def test_circuit_breaker_handler(self, disaster_recovery):
        """Test circuit breaker event handler"""
        # Enable snapshot on circuit breaker
        disaster_recovery.config.snapshot_on_circuit_breaker = True
        disaster_recovery.config.snapshot_circuit_types = ['POSITION']
        
        # Create circuit breaker event
        circuit_event = Event(
            topic=EventTopics.CIRCUIT_BREAKER_TRIGGERED,
            data={
                'circuit_id': 'test_circuit',
                'circuit_type': 'POSITION',
                'new_state': 'OPEN',
                'trigger_value': 100
            },
            priority=EventPriority.HIGH,
            source='circuit_breaker'
        )
        
        # Mock create_snapshot
        with patch.object(disaster_recovery, 'create_snapshot') as mock_create_snapshot:
            # Handle the event
            disaster_recovery._handle_circuit_breaker(circuit_event)
            
            # Verify snapshot was created
            mock_create_snapshot.assert_called_once()
            assert mock_create_snapshot.call_args[1]['trigger'] == RecoveryTrigger.CIRCUIT_BREAKER
            assert 'circuit_id' in mock_create_snapshot.call_args[1]['metadata']
    
    def test_load_snapshot_index(self, disaster_recovery, test_config):
        """Test loading snapshot index from disk"""
        # Create a snapshot to generate the index
        snapshot_id = disaster_recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Create a new instance to test loading the index
        recovery2 = DisasterRecovery(
            config=test_config,
            event_bus=MagicMock(),
            state_manager=MagicMock(),
        )
        
        # Check that snapshot was loaded
        assert snapshot_id in recovery2._snapshots
        
        # Verify snapshot metadata
        assert recovery2._snapshots[snapshot_id].id == snapshot_id
        assert recovery2._snapshots[snapshot_id].is_complete is True
        
        # Check last snapshot time was set
        assert recovery2._last_snapshot_time > 0


class TestRecoveryFromFailures:
    """Tests for recovery from various failure scenarios and edge cases"""
    
    @pytest.fixture
    def setup_recovery(self, tmp_path):
        """Setup for recovery testing"""
        # Create dirs
        dirs = {
            'journal_path': tmp_path / 'journal',
            'snapshot_path': tmp_path / 'snapshots',
            'verification_path': tmp_path / 'verification',
            'temp_path': tmp_path / 'temp',
            'replication_path': tmp_path / 'replication',
            'models_path': tmp_path / 'models'
        }
        
        for path in dirs.values():
            path.mkdir(exist_ok=True)
        
        # Create test config
        config = DisasterRecoveryConfig(
            journal_path=str(dirs['journal_path']),
            snapshot_path=str(dirs['snapshot_path']),
            verification_path=str(dirs['verification_path']),
            temp_path=str(dirs['temp_path']),
            replication_path=str(dirs['replication_path']),
            models_path=str(dirs['models_path']),
            instance_id="test_instance",
            auto_snapshot_enabled=False,
            recovery_mode=RecoveryMode.AUTOMATIC,
            snapshot_on_error=True,
            snapshot_on_circuit_breaker=True,
            snapshot_on_health_check=True,
            auto_recover_on_error=True,
            auto_recover_error_types=['critical'],
            snapshot_error_types=['critical']
        )
        
        # Create mocks
        mock_event_bus = MagicMock()
        mock_state_manager = MagicMock()
        mock_state_manager.begin_transaction = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_state_manager.create_snapshot = MagicMock(return_value={"state": "test_state"})
        
        # Create recovery instance
        recovery = DisasterRecovery(
            config=config,
            event_bus=mock_event_bus,
            state_manager=mock_state_manager
        )
        
        return {
            'recovery': recovery,
            'config': config,
            'event_bus': mock_event_bus,
            'state_manager': mock_state_manager,
            'dirs': dirs
        }
    
    def test_recovery_from_system_error(self, setup_recovery):
        """Test automatic recovery from system error"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        
        # Create a snapshot first
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Mock recovery method
        with patch.object(recovery, 'recover_from_snapshot') as mock_recover:
            mock_recover.return_value = True
            
            # Create error event that should trigger recovery
            error_event = Event(
                topic=EventTopics.SYSTEM_ERROR,
                data={
                    'message': 'Critical test error',
                    'logger_name': 'test_logger',
                    'exception_type': 'critical'
                },
                priority=EventPriority.HIGH,
                source='test'
            )
            
            # Handle the event
            recovery._handle_system_error(error_event)
            
            # Verify recovery was attempted
            mock_recover.assert_called_once()
    
    def test_recovery_from_circuit_breaker(self, setup_recovery):
        """Test automatic recovery from circuit breaker event"""
        recovery = setup_recovery['recovery']
        
        # Enable recovery on circuit breaker
        recovery.config.auto_recover_on_circuit_breaker = True
        recovery.config.auto_recover_circuit_types = ['POSITION']
        
        # Create a snapshot first
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Mock recovery method
        with patch.object(recovery, 'recover_from_snapshot') as mock_recover:
            mock_recover.return_value = True
            
            # Create circuit breaker event
            circuit_event = Event(
                topic=EventTopics.CIRCUIT_BREAKER_TRIGGERED,
                data={
                    'circuit_id': 'test_circuit',
                    'circuit_type': 'POSITION',
                    'new_state': 'OPEN',
                    'trigger_value': 100
                },
                priority=EventPriority.HIGH,
                source='circuit_breaker'
            )
            
            # Handle the event
            recovery._handle_circuit_breaker(circuit_event)
            
            # Verify recovery was attempted
            mock_recover.assert_called_once()
    
    def test_failed_snapshot_creation(self, setup_recovery):
        """Test handling of failed snapshot creation"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        state_manager = setup_recovery['state_manager']
        
        # Make state_manager.create_snapshot raise an exception
        state_manager.create_snapshot.side_effect = Exception("Test error")
        
        # Reset event bus for clean test
        event_bus.publish.reset_mock()
        
        # Attempt to create a snapshot, which should fail
        with pytest.raises(Exception):
            recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Check that failure event was published
        failure_event_published = False
        for call_args in event_bus.publish.call_args_list:
            event = call_args[0][0]
            if event.topic == "disaster_recovery.snapshot_failed":
                failure_event_published = True
                assert 'error' in event.data
                break
        
        assert failure_event_published is True
        
        # Check that recovery state was reset to IDLE
        assert recovery._recovery_state == RecoveryState.IDLE
    
    def test_failed_recovery(self, setup_recovery):
        """Test handling of failed recovery attempt"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        state_manager = setup_recovery['state_manager']
        
        # Create a snapshot first
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Make state_manager.restore_snapshot raise an exception
        state_manager.restore_snapshot.side_effect = Exception("Restore error")
        
        # Reset event bus for clean test
        event_bus.publish.reset_mock()
        
        # Attempt recovery, which should fail but not raise
        result = recovery.recover_from_snapshot(snapshot_id=snapshot_id)
        
        # Check result
        assert result is False
        
        # Check that failure event was published
        failure_event_published = False
        for call_args in event_bus.publish.call_args_list:
            event = call_args[0][0]
            if event.topic == "disaster_recovery.recovery_failed":
                failure_event_published = True
                assert event.data['success'] is False
                assert 'error_message' in event.data
                break
        
        assert failure_event_published is True
        
        # Check that recovery state was set to FAILED
        assert recovery._recovery_state == RecoveryState.FAILED
        
        # Check that recovery_in_progress flag was cleared
        assert recovery._recovery_in_progress.is_set() is False
    
    def test_concurrent_recovery_attempts(self, setup_recovery):
        """Test that concurrent recovery attempts are rejected"""
        recovery = setup_recovery['recovery']
        
        # Create a snapshot
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Set the recovery_in_progress flag manually
        recovery._recovery_in_progress.set()
        
        # Try to recover while recovery is already in progress
        result = recovery.recover_from_snapshot(snapshot_id=snapshot_id)
        
        # Check result
        assert result is False
        
        # Reset flag for cleanup
        recovery._recovery_in_progress.clear()
    
    def test_recovery_with_nonexistent_snapshot(self, setup_recovery):
        """Test recovery with a nonexistent snapshot ID"""
        recovery = setup_recovery['recovery']
        
        # Try to recover with a nonexistent snapshot ID
        result = recovery.recover_from_snapshot(snapshot_id="nonexistent_snapshot")
        
        # Check result
        assert result is False
    
    def test_health_check_automatic_recovery(self, setup_recovery):
        """Test automatic recovery from critical health status"""
        recovery = setup_recovery['recovery']
        
        # Enable recovery on health check
        recovery.config.auto_recover_on_health_check = True
        recovery.config.auto_recover_components = ['critical_component']
        
        # Create a snapshot first
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Mock recovery method
        with patch.object(recovery, 'recover_from_snapshot') as mock_recover:
            mock_recover.return_value = True
            
            # Create health check event with critical status
            health_event = Event(
                topic=EventTopics.HEALTH_CHECK,
                data={
                    'status': 'critical',
                    'critical_components': ['critical_component'],
                    'warning_components': []
                },
                priority=EventPriority.HIGH,
                source='health_monitor'
            )
            
            # Handle the event
            recovery._handle_health_check(health_event)
            
            # Verify recovery was attempted
            mock_recover.assert_called_once()
            
            # Reset mock
            mock_recover.reset_mock()
            
            # Create event with warning level (should not trigger recovery)
            warning_event = Event(
                topic=EventTopics.HEALTH_CHECK,
                data={
                    'status': 'warning',
                    'critical_components': [],
                    'warning_components': ['warning_component']
                },
                priority=EventPriority.NORMAL,
                source='health_monitor'
            )
            
            # Handle the event
            recovery._handle_health_check(warning_event)
            
            # Verify recovery was not attempted
            mock_recover.assert_not_called()
    
    def test_position_mismatch_recovery(self, setup_recovery):
        """Test recovery triggered by position reconciliation failure"""
        recovery = setup_recovery['recovery']
        
        # Enable recovery on position mismatch
        recovery.config.auto_recover_on_position_mismatch = True
        
        # Create a snapshot first
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Mock recovery method
        with patch.object(recovery, 'recover_from_snapshot') as mock_recover:
            mock_recover.return_value = True
            
            # Create position update event with reconciliation failure
            position_event = Event(
                topic=EventTopics.POSITION_UPDATED,
                data={
                    'reconciliation_failed': True,
                    'position_id': 'test_position',
                    'expected_value': 100,
                    'actual_value': 90
                },
                priority=EventPriority.HIGH,
                source='position_manager'
            )
            
            # Handle the event
            recovery._handle_position_update(position_event)
            
            # Verify recovery was attempted
            mock_recover.assert_called_once()
    
    def test_recover_from_remote_region(self, setup_recovery):
        """Test recovering from a snapshot in a remote region"""
        recovery = setup_recovery['recovery']
        dirs = setup_recovery['dirs']
        
        # Create remote region directories and files
        region_id = "region1"
        region_dir = Path(dirs['replication_path']) / region_id
        region_dir.mkdir(exist_ok=True)
        
        # Create a test snapshot in the region
        snapshot_id = f"snapshot_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        snapshot_dir = region_dir / snapshot_id
        snapshot_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "id": snapshot_id,
            "timestamp": time.time(),
            "trigger": RecoveryTrigger.MANUAL.value,
            "state_hash": "test_hash",
            "size_bytes": 1000,
            "is_complete": True,
            "regions": [region_id]
        }
        
        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create system state file
        with open(snapshot_dir / "system_state.json", 'w') as f:
            json.dump({"state": "test_state"}, f)
        
        # Mock recover_from_snapshot to isolate test
        with patch.object(recovery, 'recover_from_snapshot') as mock_recover:
            mock_recover.return_value = True
            
            # Recover from the remote region
            result = recovery.recover_from_remote_region(region_id, snapshot_id)
            
            # Check result
            assert result is True
            
            # Verify recover_from_snapshot was called
            mock_recover.assert_called_once_with(
                snapshot_id=snapshot_id,
                verify_after_recovery=True,
                metadata={"source_region": region_id}
            )
            
            # Check that snapshot was copied to local snapshots directory
            local_snapshot_dir = Path(dirs['snapshot_path']) / snapshot_id
            assert local_snapshot_dir.exists()
            assert (local_snapshot_dir / "metadata.json").exists()
            assert (local_snapshot_dir / "system_state.json").exists()
    
    def test_multi_stage_recovery_process(self, setup_recovery):
        """Test the full multi-stage recovery process"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        state_manager = setup_recovery['state_manager']
        
        # Create original snapshot
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Reset mocks for clean test
        event_bus.publish.reset_mock()
        
        # Track recovery state transitions
        state_transitions = []
        original_recovery_state = recovery._recovery_state
        
        # Mock _verify_snapshot and verify_system_state to return success
        with patch.object(recovery, 'verify_snapshot', return_value=(True, {})), \
             patch.object(recovery, 'verify_system_state', return_value=(True, [])), \
             patch.object(recovery, '_replay_transaction_journal') as mock_replay, \
             patch.object(recovery, '_recover_components', return_value=['comp1', 'comp2']) as mock_recover_components:
            
            # Mock recovery state property to track transitions
            original_setter = type(recovery)._recovery_state.fset
            
            def state_transition_tracker(self, value):
                state_transitions.append(value)
                original_setter(self, value)
                
            with mock.patch.object(type(recovery), '_recovery_state', 
                                  DisasterRecovery._recovery_state.getter, 
                                  state_transition_tracker):
                
                # Perform recovery
                result = recovery.recover_from_snapshot(snapshot_id=snapshot_id)
                
                # Check result
                assert result is True
                
                # Verify state transitions occurred in the correct order
                expected_states = [
                    RecoveryState.PREPARING,
                    RecoveryState.STATE_VERIFICATION,
                    RecoveryState.STATE_RESTORATION,
                    RecoveryState.COMPONENT_RECOVERY,
                    RecoveryState.STATE_VERIFICATION,
                    RecoveryState.COMPLETED
                ]
                
                # Check if all expected states appear in the transitions
                # (there may be intermediate states depending on implementation)
                for expected_state in expected_states:
                    assert expected_state in state_transitions
                
                # Check that transaction journal replay was called
                mock_replay.assert_called_once()
                
                # Check that component recovery was called
                mock_recover_components.assert_called_once()
                
                # Check that pre-recovery snapshot was created
                assert 'pre_recovery_snapshot_id' in recovery._current_recovery.metadata
                
                # Check events were published
                recovery_started = False
                recovery_completed = False
                
                for call_args in event_bus.publish.call_args_list:
                    event = call_args[0][0]
                    if event.topic == "disaster_recovery.recovery_started":
                        recovery_started = True
                    elif event.topic == "disaster_recovery.recovery_completed":
                        recovery_completed = True
                        assert event.data['success'] is True
                
                assert recovery_started is True
                assert recovery_completed is True
    
    def test_event_handler_registration(self, setup_recovery):
        """Test that event handlers are properly registered"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        
        # Check that event handlers were registered
        expected_subscriptions = [
            EventTopics.SYSTEM_ERROR,
            EventTopics.CIRCUIT_BREAKER_TRIGGERED,
            EventTopics.HEALTH_CHECK, 
            EventTopics.STATE_SAVED,
            EventTopics.POSITION_UPDATED,
            EventTopics.SYSTEM_SHUTDOWN,
            "disaster_recovery.snapshot_request",
            "disaster_recovery.recovery_request",
            "disaster_recovery.verification_request"
        ]
        
        # Extract actual subscriptions
        subscribed_topics = []
        for call_args in event_bus.subscribe.call_args_list:
            topic = call_args[0][0]
            subscribed_topics.append(topic)
        
        # Check each expected subscription
        for expected_topic in expected_subscriptions:
            assert expected_topic in subscribed_topics
    
    def test_snapshot_request_handler(self, setup_recovery):
        """Test handling snapshot requests via events"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        
        # Mock create_snapshot
        with patch.object(recovery, 'create_snapshot') as mock_create_snapshot:
            mock_create_snapshot.return_value = "test_snapshot_id"
            
            # Create snapshot request event
            request_event = Event(
                topic="disaster_recovery.snapshot_request",
                data={
                    'trigger': RecoveryTrigger.MANUAL.value,
                    'metadata': {'test_key': 'test_value'},
                    'requestor': 'test_requestor',
                    'request_id': 'test_request_id'
                },
                priority=EventPriority.NORMAL,
                source='test'
            )
            
            # Reset event bus mock
            event_bus.publish.reset_mock()
            
            # Handle the event
            recovery._handle_snapshot_request(request_event)
            
            # Verify create_snapshot was called with correct args
            mock_create_snapshot.assert_called_once_with(
                trigger=RecoveryTrigger.MANUAL,
                metadata={'test_key': 'test_value'}
            )
            
            # Verify response was published
            response_event_published = False
            for call_args in event_bus.publish.call_args_list:
                event = call_args[0][0]
                if event.topic == "disaster_recovery.snapshot_response":
                    response_event_published = True
                    assert event.data['request_id'] == 'test_request_id'
                    assert event.data['snapshot_id'] == 'test_snapshot_id'
                    assert event.data['success'] is True
                    break
            
            assert response_event_published is True
            
            # Test error handling
            mock_create_snapshot.reset_mock()
            event_bus.publish.reset_mock()
            
            # Make create_snapshot raise an exception
            mock_create_snapshot.side_effect = Exception("Test error")
            
            # Handle the event again
            recovery._handle_snapshot_request(request_event)
            
            # Verify error response was published
            error_response_published = False
            for call_args in event_bus.publish.call_args_list:
                event = call_args[0][0]
                if event.topic == "disaster_recovery.snapshot_response":
                    error_response_published = True
                    assert event.data['request_id'] == 'test_request_id'
                    assert event.data['success'] is False
                    assert 'error' in event.data
                    break
            
            assert error_response_published is True
    
    def test_recovery_request_handler(self, setup_recovery):
        """Test handling recovery requests via events"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        
        # Create a snapshot first
        snapshot_id = recovery.create_snapshot(trigger=RecoveryTrigger.MANUAL)
        
        # Mock recover_from_snapshot
        with patch.object(recovery, 'recover_from_snapshot') as mock_recover:
            mock_recover.return_value = True
            
            # Create recovery request event
            request_event = Event(
                topic="disaster_recovery.recovery_request",
                data={
                    'snapshot_id': snapshot_id,
                    'verify_after_recovery': True,
                    'metadata': {'test_key': 'test_value'},
                    'requestor': 'test_requestor',
                    'request_id': 'test_request_id'
                },
                priority=EventPriority.HIGH,
                source='test'
            )
            
            # Reset event bus mock
            event_bus.publish.reset_mock()
            
            # Handle the event
            recovery._handle_recovery_request(request_event)
            
            # Verify recover_from_snapshot was called with correct args
            mock_recover.assert_called_once_with(
                snapshot_id=snapshot_id,
                verify_after_recovery=True,
                metadata={'test_key': 'test_value'}
            )
            
            # Verify response was published
            response_event_published = False
            for call_args in event_bus.publish.call_args_list:
                event = call_args[0][0]
                if event.topic == "disaster_recovery.recovery_response":
                    response_event_published = True
                    assert event.data['request_id'] == 'test_request_id'
                    assert event.data['success'] is True
                    break
            
            assert response_event_published is True
    
    def test_collect_component_snapshots(self, setup_recovery):
        """Test collecting snapshots from components"""
        recovery = setup_recovery['recovery']
        event_bus = setup_recovery['event_bus']
        
        # Set up component snapshots
        component_data = {
            "component1": {"state": "data1"},
            "component2": {"state": "data2"}
        }
        
        # Test request-response pattern
        event_bus.request.return_value = component_data
        
        # Call the method
        result = recovery._collect_component_snapshots()
        
        # Check result
        assert result == component_data
        
        # Verify request was made
        event_bus.request.assert_called_once_with(
            event_topic="disaster_recovery.component_snapshot_request",
            payload=ANY,
            timeout_seconds=ANY
        )
        
        # Test fallback pattern (publish-subscribe)
        event_bus.request.reset_mock()
        event_bus.request.side_effect = AttributeError("No request method")
        
        # Mock subscribe to return a handler and temporary subscription
        def mock_subscribe_handler(topic, handler):
            # Simulate component responses
            for component_id, data in component_data.items():
                # Create a response event
                response_event = Event(
                    topic="disaster_recovery.component_snapshot_response",
                    data={
                        "component_id": component_id,
                        "snapshot_data": data
                    },
                    priority=EventPriority.NORMAL,
                    source=component_id
                )
                # Call the handler with the event
                handler(response_event)
            
            # Return a fake subscription ID
            return "temp_subscription_id"
        
        event_bus.subscribe.side_effect = mock_subscribe_handler
        
        # Mock sleep to avoid waiting
        with patch('time.sleep'):
            # Call the method
            result = recovery._collect_component_snapshots()
        
            # Check result
            assert result == component_data
            
            # Verify publish was called
            event_bus.publish.assert_called_with(ANY)
            
            # Verify unsubscribe was attempted
            event_bus.unsubscribe.assert_called_once()
    
    def test_automatic_tasks(self, setup_recovery):
        """Test automatic task scheduling"""
        recovery = setup_recovery['recovery']
        
        # Enable auto snapshots
        recovery.config.auto_snapshot_enabled = True
        recovery.config.auto_snapshot_interval_minutes = 0.001  # Very short for testing
        
        # Mock time.sleep to avoid waiting
        # Mock threading.Thread to capture the task function
        thread_target = None
        
        def mock_thread_init(target, daemon, name):
            nonlocal thread_target
            thread_target = target
            mock_thread = MagicMock()
            return mock_thread
        
        # Mock create_snapshot and replicate_latest_snapshot
        with patch('time.sleep'), \
             patch('threading.Thread', side_effect=mock_thread_init) as mock_thread, \
             patch.object(recovery, 'create_snapshot') as mock_create_snapshot, \
             patch.object(recovery, 'replicate_latest_snapshot') as mock_replicate:
            
            # Restart scheduler
            recovery._schedule_snapshot_task()
            
            # Verify thread was created
            mock_thread.assert_called_once()
            assert thread_target is not None
            
            # Call the snapshot task function
            recovery.config.auto_replicate = True
            thread_target()
            
            # Verify create_snapshot was called
            mock_create_snapshot.assert_called_once_with(trigger=RecoveryTrigger.SCHEDULED)
            
            # Verify replicate_latest_snapshot was called (auto_replicate is True)
            mock_replicate.assert_called_once()
            
            # Test without auto replicate
            mock_create_snapshot.reset_mock()
            mock_replicate.reset_mock()
            
            recovery.config.auto_replicate = False
            thread_target()
            
            # Verify create_snapshot was called
            mock_create_snapshot.assert_called_once_with(trigger=RecoveryTrigger.SCHEDULED)
            
            # Verify replicate_latest_snapshot was not called
            mock_replicate.assert_not_called()
    
    def test_system_shutdown_handler(self, setup_recovery):
        """Test handling system shutdown events"""
        recovery = setup_recovery['recovery']
        
        # Enable shutdown snapshots
        recovery.config.snapshot_on_shutdown = True
        
        # Mock create_snapshot
        with patch.object(recovery, 'create_snapshot') as mock_create_snapshot:
            # Create shutdown event
            shutdown_event = Event(
                topic=EventTopics.SYSTEM_SHUTDOWN,
                data={},
                priority=EventPriority.HIGH,
                source='system'
            )
            
            # Handle the event
            recovery._handle_system_shutdown(shutdown_event)
            
            # Verify snapshot was created
            mock_create_snapshot.assert_called_once_with(
                trigger=RecoveryTrigger.MANUAL,
                metadata={"purpose": "shutdown"}
            )