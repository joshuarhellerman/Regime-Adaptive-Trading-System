import pytest
import time
import threading
from unittest.mock import MagicMock, patch
import networkx as nx

from core.component_registry import ComponentRegistry, ComponentStatus


class TestComponent:
    """Test component for registry tests"""
    version = "1.0.0"
    description = "Test component for unit tests"

    def __init__(self, name):
        self.name = name
        self.started = False
        self.stopped = False
        self.error_on_start = False
        self.error_on_stop = False
        self.start_return_value = None
        self.stop_return_value = None
        self.start_call_count = 0
        self.stop_call_count = 0

    def start(self):
        """Start the component"""
        self.start_call_count += 1
        self.started = True
        if self.error_on_start:
            raise Exception("Test exception on start")
        return self.start_return_value

    def stop(self):
        """Stop the component"""
        self.stop_call_count += 1
        self.stopped = True
        if self.error_on_stop:
            raise Exception("Test exception on stop")
        return self.stop_return_value

    def get_status(self):
        """Get component status"""
        return {
            "started": self.started,
            "custom_field": "test value"
        }

    def execute_command(self, command, params=None):
        """Execute a command"""
        params = params or {}
        if command == "test_command":
            return {"success": True, "result": "command executed"}
        elif command == "error_command":
            raise Exception("Test exception on command")
        return {"success": False, "error": "Unknown command"}


class NoStartStopComponent:
    """Test component without start/stop methods"""
    def __init__(self, name):
        self.name = name


@pytest.fixture
def event_bus_mock():
    """Create a mock event bus"""
    mock = MagicMock()
    mock.publish = MagicMock()
    return mock


@pytest.fixture
def registry(event_bus_mock):
    """Create a test component registry with mocked event bus"""
    with patch('core.component_registry.get_event_bus', return_value=event_bus_mock):
        registry = ComponentRegistry()
        yield registry


def test_register_component(registry):
    """Test basic component registration"""
    component = TestComponent("test1")
    
    # Register component
    result = registry.register(component, "test1", "test_type")
    
    # Verify registration
    assert result is True
    assert registry.get("test1") is component
    assert registry.get_status("test1") == ComponentStatus.REGISTERED
    
    # Check component info
    info = registry.get_component_info("test1")
    assert info["name"] == "test1"
    assert info["type"] == "test_type"
    assert info["class"] == "TestComponent"
    assert info["has_start"] is True
    assert info["has_stop"] is True
    assert info["has_status"] is True
    assert "methods" in info


def test_register_duplicate_component(registry):
    """Test registering a component with a duplicate name"""
    component1 = TestComponent("test1")
    component2 = TestComponent("test1")
    
    registry.register(component1, "test1")
    
    # Attempt to register duplicate
    with pytest.raises(ValueError, match="already registered"):
        registry.register(component2, "test1")


def test_register_with_dependencies(registry):
    """Test registering components with dependencies"""
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2")
    
    # Register components with dependencies
    registry.register(component1, "comp1")
    registry.register(component2, "comp2", dependencies=["comp1"])
    
    # Check dependency tracking
    assert registry._dependencies["comp2"] == ["comp1"]
    assert registry._dependents["comp1"] == ["comp2"]


def test_register_with_nonexistent_dependencies(registry):
    """Test registering with dependencies that don't exist yet"""
    component = TestComponent("test1")
    
    # Register with nonexistent dependency
    registry.register(component, "test1", dependencies=["not_registered_yet"])
    
    # Check dependency tracking
    assert registry._dependencies["test1"] == ["not_registered_yet"]
    assert registry._dependents["not_registered_yet"] == ["test1"]


def test_unregister_component(registry):
    """Test unregistering a component"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Unregister
    result = registry.unregister("test1")
    
    # Verify unregistration
    assert result is True
    assert registry.get("test1") is None
    assert "test1" not in registry._components


def test_unregister_running_component(registry):
    """Test attempting to unregister a running component"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Start component
    registry.start("test1")
    
    # Attempt to unregister while running
    result = registry.unregister("test1")
    
    # Verify unregistration failed
    assert result is False
    assert registry.get("test1") is component


def test_unregister_with_dependents(registry):
    """Test attempting to unregister a component with dependents"""
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2")
    
    registry.register(component1, "comp1")
    registry.register(component2, "comp2", dependencies=["comp1"])
    
    # Attempt to unregister dependency
    result = registry.unregister("comp1")
    
    # Verify unregistration failed
    assert result is False
    assert registry.get("comp1") is component1


def test_start_component(registry):
    """Test starting a component"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Start component
    result = registry.start("test1")
    
    # Verify start
    assert result is True
    assert component.started is True
    assert registry.get_status("test1") == ComponentStatus.RUNNING
    
    # Check component info includes runtime info
    info = registry.get_component_info("test1")
    assert "start_time" in info
    assert "uptime" in info
    assert "runtime" in info


def test_start_component_with_dependencies(registry):
    """Test starting a component with dependencies"""
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2")
    
    registry.register(component1, "comp1")
    registry.register(component2, "comp2", dependencies=["comp1"])
    
    # Start component with dependency
    result = registry.start("comp2")
    
    # Verify both components started
    assert result is True
    assert component1.started is True
    assert component2.started is True
    assert registry.get_status("comp1") == ComponentStatus.RUNNING
    assert registry.get_status("comp2") == ComponentStatus.RUNNING


def test_start_component_explicit_false_return(registry):
    """Test starting a component that returns False from start method"""
    component = TestComponent("test1")
    component.start_return_value = False
    registry.register(component, "test1")
    
    # Start component
    result = registry.start("test1")
    
    # Verify start failed
    assert result is False
    assert component.started is True  # The method was still called
    assert registry.get_status("test1") == ComponentStatus.ERROR


def test_start_component_exception(registry):
    """Test starting a component that raises an exception"""
    component = TestComponent("test1")
    component.error_on_start = True
    registry.register(component, "test1")
    
    # Start component
    result = registry.start("test1")
    
    # Verify start failed
    assert result is False
    assert component.start_call_count == 1
    assert registry.get_status("test1") == ComponentStatus.ERROR


def test_start_component_no_start_method(registry):
    """Test starting a component without a start method"""
    component = NoStartStopComponent("test1")
    registry.register(component, "test1")
    
    # Start component
    result = registry.start("test1")
    
    # Verify component is considered running
    assert result is True
    assert registry.get_status("test1") == ComponentStatus.RUNNING


def test_stop_component(registry):
    """Test stopping a component"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Start then stop
    registry.start("test1")
    result = registry.stop("test1")
    
    # Verify stop
    assert result is True
    assert component.stopped is True
    assert registry.get_status("test1") == ComponentStatus.STOPPED


def test_stop_component_with_dependents(registry):
    """Test stopping a component with dependents"""
    component1 = TestComponent("comp1")
    component2 = TestComponent("comp2")
    
    registry.register(component1, "comp1")
    registry.register(component2, "comp2", dependencies=["comp1"])
    
    # Start both components
    registry.start("comp2")  # This will also start comp1
    
    # Stop dependency (should stop dependents first)
    result = registry.stop("comp1")
    
    # Verify both stopped
    assert result is True
    assert component1.stopped is True
    assert component2.stopped is True
    assert registry.get_status("comp1") == ComponentStatus.STOPPED
    assert registry.get_status("comp2") == ComponentStatus.STOPPED


def test_stop_component_explicit_false_return(registry):
    """Test stopping a component that returns False from stop method"""
    component = TestComponent("test1")
    component.stop_return_value = False
    registry.register(component, "test1")
    
    # Start then stop
    registry.start("test1")
    result = registry.stop("test1")
    
    # Verify stop failed
    assert result is False
    assert component.stopped is True  # The method was still called
    assert registry.get_status("test1") == ComponentStatus.ERROR


def test_stop_component_exception(registry):
    """Test stopping a component that raises an exception"""
    component = TestComponent("test1")
    component.error_on_stop = True
    registry.register(component, "test1")
    
    # Start then stop
    registry.start("test1")
    result = registry.stop("test1")
    
    # Verify stop failed
    assert result is False
    assert component.stop_call_count == 1
    assert registry.get_status("test1") == ComponentStatus.ERROR


def test_stop_component_no_stop_method(registry):
    """Test stopping a component without a stop method"""
    component = NoStartStopComponent("test1")
    registry.register(component, "test1")
    
    # Mark as running then stop
    registry._status["test1"] = ComponentStatus.RUNNING
    result = registry.stop("test1")
    
    # Verify stop succeeded
    assert result is True
    assert registry.get_status("test1") == ComponentStatus.STOPPED


def test_start_all(registry):
    """Test starting all components"""
    components = [
        TestComponent("comp1"),
        TestComponent("comp2"),
        TestComponent("comp3")
    ]
    
    # Register components with dependencies
    registry.register(components[0], "comp1")
    registry.register(components[1], "comp2", dependencies=["comp1"])
    registry.register(components[2], "comp3", dependencies=["comp2"])
    
    # Start all
    result = registry.start_all()
    
    # Verify all started
    assert result is True
    assert all(c.started for c in components)
    
    # Verify startup complete flag
    assert registry._startup_complete is True


def test_start_all_with_failures(registry):
    """Test starting all components when some fail"""
    components = [
        TestComponent("comp1"),
        TestComponent("comp2"),
        TestComponent("comp3")
    ]
    
    # Make one component fail
    components[1].error_on_start = True
    
    # Register components
    registry.register(components[0], "comp1")
    registry.register(components[1], "comp2")
    registry.register(components[2], "comp3")
    
    # Start all
    result = registry.start_all()
    
    # Verify overall failure but some components started
    assert result is False
    assert components[0].started is True
    assert components[1].start_call_count == 1
    assert components[2].started is True  # Should still try to start other components


def test_stop_all(registry):
    """Test stopping all components"""
    components = [
        TestComponent("comp1"),
        TestComponent("comp2"),
        TestComponent("comp3")
    ]
    
    # Register components with dependencies
    registry.register(components[0], "comp1")
    registry.register(components[1], "comp2", dependencies=["comp1"])
    registry.register(components[2], "comp3", dependencies=["comp2"])
    
    # Start all then stop all
    registry.start_all()
    result = registry.stop_all()
    
    # Verify all stopped
    assert result is True
    assert all(c.stopped for c in components)
    
    # Verify shutdown flag
    assert registry._shutdown_in_progress is False


def test_stop_all_with_failures(registry):
    """Test stopping all components when some fail"""
    components = [
        TestComponent("comp1"),
        TestComponent("comp2"),
        TestComponent("comp3")
    ]
    
    # Make one component fail on stop
    components[1].error_on_stop = True
    
    # Register and start components
    for i, comp in enumerate(components):
        registry.register(comp, f"comp{i+1}")
        registry._status[f"comp{i+1}"] = ComponentStatus.RUNNING
    
    # Stop all
    result = registry.stop_all()
    
    # Verify overall failure but attempted to stop all
    assert result is False
    assert components[0].stopped is True
    assert components[1].stop_call_count == 1
    assert components[2].stopped is True


def test_restart_component(registry):
    """Test restarting a component"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Start, then restart
    registry.start("test1")
    result = registry.restart_component("test1")
    
    # Verify restart success
    assert result is True
    assert component.stop_call_count == 1
    assert component.start_call_count == 2
    assert registry.get_status("test1") == ComponentStatus.RUNNING


def test_restart_all(registry):
    """Test restarting all components"""
    components = [
        TestComponent("comp1"),
        TestComponent("comp2")
    ]
    
    # Register components
    registry.register(components[0], "comp1")
    registry.register(components[1], "comp2", dependencies=["comp1"])
    
    # Start all, then restart all
    registry.start_all()
    
    # Reset counters after initial start
    for c in components:
        c.start_call_count = 0
        c.stop_call_count = 0
    
    result = registry.restart_all()
    
    # Verify restart success
    assert result is True
    assert all(c.stop_call_count == 1 for c in components)
    assert all(c.start_call_count == 1 for c in components)


def test_execute_command(registry):
    """Test executing a command on a component"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Execute command
    result = registry.execute_command("test1", "test_command", {"param": "value"})
    
    # Verify command execution
    assert result["success"] is True
    assert result["result"] == "command executed"


def test_execute_command_error(registry):
    """Test executing a command that raises an exception"""
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Execute command that raises exception
    result = registry.execute_command("test1", "error_command")
    
    # Verify error handling
    assert result["success"] is False
    assert "error" in result


def test_execute_command_unknown_component(registry):
    """Test executing a command on unknown component"""
    result = registry.execute_command("unknown", "test_command")
    
    # Verify error handling
    assert result["success"] is False
    assert "not found" in result["error"]


def test_execute_command_no_command_support(registry):
    """Test executing a command on component without command support"""
    component = NoStartStopComponent("test1")
    registry.register(component, "test1")
    
    # Execute command
    result = registry.execute_command("test1", "test_command")
    
    # Verify error handling
    assert result["success"] is False
    assert "does not support commands" in result["error"]


def test_get_dependency_graph(registry):
    """Test getting the dependency graph representation"""
    # Register components with dependencies
    registry.register(TestComponent("comp1"), "comp1")
    registry.register(TestComponent("comp2"), "comp2", dependencies=["comp1"])
    registry.register(TestComponent("comp3"), "comp3", dependencies=["comp2"])
    
    # Get dependency graph
    graph = registry.get_dependency_graph()
    
    # Verify graph structure
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2
    
    # Verify node structure
    node_ids = [node["id"] for node in graph["nodes"]]
    assert "comp1" in node_ids
    assert "comp2" in node_ids
    assert "comp3" in node_ids
    
    # Verify edge structure
    edge_pairs = [(edge["from"], edge["to"]) for edge in graph["edges"]]
    assert ("comp1", "comp2") in edge_pairs
    assert ("comp2", "comp3") in edge_pairs


def test_get_dependency_ordering(registry):
    """Test getting components ordered by dependency level"""
    # Register components with dependencies
    registry.register(TestComponent("comp1"), "comp1")
    registry.register(TestComponent("comp2"), "comp2", dependencies=["comp1"])
    registry.register(TestComponent("comp3"), "comp3", dependencies=["comp2"])
    registry.register(TestComponent("comp4"), "comp4", dependencies=["comp1"])
    
    # Get dependency ordering
    ordering = registry._get_dependency_ordering()
    
    # Verify ordering (should be a list of lists)
    assert len(ordering) >= 1
    
    # First level should contain only comp1
    assert "comp1" in ordering[0]
    
    # Second level should contain comp2 and comp4
    if len(ordering) > 1:
        assert "comp2" in ordering[1] or "comp4" in ordering[1]
    
    # Verify comp3 comes after comp2
    comp2_level = None
    comp3_level = None
    for i, level in enumerate(ordering):
        if "comp2" in level:
            comp2_level = i
        if "comp3" in level:
            comp3_level = i
    
    assert comp2_level is not None
    assert comp3_level is not None
    assert comp3_level > comp2_level


def test_circular_dependency_detection(registry):
    """Test detection of circular dependencies"""
    # Register components with circular dependencies
    registry.register(TestComponent("comp1"), "comp1")
    registry.register(TestComponent("comp2"), "comp2", dependencies=["comp1"])
    registry.register(TestComponent("comp3"), "comp3", dependencies=["comp2"])
    
    # Add circular dependency
    registry._dependencies["comp1"] = ["comp3"]
    registry._dependents["comp3"].append("comp1")
    
    # Get dependency ordering (should detect cycles)
    with pytest.raises(ValueError, match="Circular dependencies detected"):
        registry._get_dependency_ordering()


def test_is_dependency_satisfied(registry):
    """Test checking if a dependency is satisfied"""
    # Register components
    registry.register(TestComponent("comp1"), "comp1")
    registry.register(TestComponent("comp2"), "comp2", dependencies=["comp1"])
    
    # Initially not satisfied
    assert registry.is_dependency_satisfied("comp2", "comp1") is False
    
    # Start dependency
    registry.start("comp1")
    
    # Now satisfied
    assert registry.is_dependency_satisfied("comp2", "comp1") is True
    
    # Unknown dependency
    assert registry.is_dependency_satisfied("comp2", "unknown") is False


def test_get_all_component_info(registry):
    """Test getting info for all components"""
    # Register components
    registry.register(TestComponent("comp1"), "comp1")
    registry.register(TestComponent("comp2"), "comp2")
    
    # Get all info
    info = registry.get_all_component_info()
    
    # Verify info
    assert "comp1" in info
    assert "comp2" in info
    assert info["comp1"]["name"] == "comp1"
    assert info["comp2"]["name"] == "comp2"


def test_get_component_names(registry):
    """Test getting all component names"""
    # Register components
    registry.register(TestComponent("comp1"), "comp1")
    registry.register(TestComponent("comp2"), "comp2")
    
    # Get names
    names = registry.get_component_names()
    
    # Verify names
    assert "comp1" in names
    assert "comp2" in names
    assert len(names) == 2


def test_component_event_publishing(registry, event_bus_mock):
    """Test that events are published for component lifecycle changes"""
    # Register component
    component = TestComponent("test1")
    registry.register(component, "test1")
    
    # Start component
    registry.start("test1")
    
    # Stop component
    registry.stop("test1")
    
    # Verify events were published
    assert event_bus_mock.publish.call_count >= 3  # At least register, start, stop events