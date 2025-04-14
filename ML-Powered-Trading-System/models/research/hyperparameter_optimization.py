# Clone estimator
            model = self._clone_estimator(estimator)
            
            # Set parameters
            for param, value in params.items():
                setattr(model, param, value)
            
            # Evaluate on CV splits
            scores = []
            for split in cv_splits:
                # Get train/test data
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Score model
                score = self._compute_score(model, X_test, y_test)
                scores.append(score)
            
            # Compute mean score
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            return {
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": std_score,
                "scores": scores
            }
        
        # Use parallel processing if n_jobs > 1
        if self.n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(evaluate_params, iterator))
                all_results.extend(results)
        else:
            # Serial processing
            for params in iterator:
                result = evaluate_params(params)
                all_results.append(result)
                
                # Update best score
                if result["mean_test_score"] > best_score:
                    best_score = result["mean_test_score"]
                    best_params = result["params"]
                    
                    if self.verbose > 1:
                        logger.info(f"New best score: {best_score:.4f} with parameters: {best_params}")
                
                # Report progress
                if self.progress_callback:
                    progress = len(all_results) / self.n_iterations
                    self.progress_callback(
                        progress, 
                        {"current_score": result["mean_test_score"]}, 
                        {"best_score": best_score, "best_params": best_params}
                    )
        
        # Sort results by mean score
        all_results = sorted(all_results, key=lambda x: x["mean_test_score"], reverse=True)
        
        # Get best parameters and score
        best_result = all_results[0]
        best_params = best_result["params"]
        best_score = best_result["mean_test_score"]
        cv_scores = best_result["scores"]
        
        # Add rank to results
        for i, result in enumerate(all_results):
            result["rank"] = i + 1
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,  # Will be updated later
            method=OptimizationMethod.RANDOM_SEARCH,
            model_type=estimator.__class__.__name__,
            cv_scores=cv_scores
        )
    
    def _bayesian_optimization(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform Bayesian optimization.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        if not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available. Using random search instead.")
            return self._random_search(estimator, params_dict, X, y)
        
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        
        # Convert parameters to search space
        search_spaces = {}
        for param_name, param_info in params_dict.items():
            if param_info["type"] in ["categorical", "boolean"]:
                search_spaces[param_name] = Categorical(param_info["values"])
            elif param_info["type"] == "float":
                range_min, range_max = param_info["range"]
                if param_info["log_scale"]:
                    if range_min <= 0:
                        range_min = 1e-6  # Avoid non-positive values for log scale
                    search_spaces[param_name] = Real(range_min, range_max, prior="log-uniform")
                else:
                    search_spaces[param_name] = Real(range_min, range_max, prior="uniform")
            elif param_info["type"] == "int":
                range_min, range_max = param_info["range"]
                if param_info["log_scale"]:
                    if range_min <= 0:
                        range_min = 1  # Avoid non-positive values for log scale
                    search_spaces[param_name] = Integer(range_min, range_max, prior="log-uniform")
                else:
                    search_spaces[param_name] = Integer(range_min, range_max, prior="uniform")
        
        # Create BayesSearchCV estimator
        opt = BayesSearchCV(
            estimator,
            search_spaces,
            n_iter=self.n_iterations,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=False,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # Fit the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.fit(X, y)
        
        # Get results
        best_params = opt.best_params_
        best_score = opt.best_score_
        
        # Format all results
        all_results = []
        for i, params in enumerate(opt.cv_results_["params"]):
            result = {
                "params": params,
                "mean_test_score": opt.cv_results_["mean_test_score"][i],
                "std_test_score": opt.cv_results_["std_test_score"][i],
                "rank": i + 1
            }
            all_results.append(result)
        
        # Sort by mean test score (descending)
        all_results = sorted(all_results, key=lambda x: x["mean_test_score"], reverse=True)
        
        # Get CV scores for best parameters
        best_idx = opt.best_index_
        cv_scores = opt.cv_results_["split_test_scores"][best_idx]
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,  # Will be updated later
            method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            model_type=estimator.__class__.__name__,
            cv_scores=cv_scores,
            metadata={"optimizer": opt}
        )
    
    def _tpe_optimization(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform TPE (Tree-structured Parzen Estimator) optimization using Optuna.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("optuna not available. Using random search instead.")
            return self._random_search(estimator, params_dict, X, y)
        
        import optuna
        from optuna.samplers import TPESampler
        
        # Get CV splits
        cv_splits = self.cv.split(X, y)
        
        # Define the objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_info in params_dict.items():
                if param_info["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_info["values"])
                elif param_info["type"] == "boolean":
                    params[param_name] = trial.suggest_categorical(param_name, [True, False])
                elif param_info["type"] == "float":
                    range_min, range_max = param_info["range"]
                    if param_info["log_scale"]:
                        if range_min <= 0:
                            range_min = 1e-6
                        params[param_name] = trial.suggest_float(param_name, range_min, range_max, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, range_min, range_max)
                elif param_info["type"] == "int":
                    range_min, range_max = param_info["range"]
                    if param_info["log_scale"]:
                        if range_min <= 0:
                            range_min = 1
                        params[param_name] = trial.suggest_int(param_name, range_min, range_max, log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, range_min, range_max)
            
            # Clone estimator
            model = self._clone_estimator(estimator)
            
            # Set parameters
            for param, value in params.items():
                setattr(model, param, value)
            
            # Evaluate on CV splits
            scores = []
            for split in cv_splits:
                # Get train/test data
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Score model
                score = self._compute_score(model, X_test, y_test)
                scores.append(score)
            
            # Return mean score across splits
            return np.mean(scores)
        
        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_iterations,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose > 0
        )
        
        # Get best parameters and score
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Format all results
        all_results = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    "params": trial.params,
                    "mean_test_score": trial.value,
                    "std_test_score": 0.0,  # Not available from Optuna
                    "rank": i + 1
                }
                all_results.append(result)
        
        # Sort by mean test score (descending)
        all_results = sorted(all_results, key=lambda x: x["mean_test_score"], reverse=True)
        
        # Add rank to results
        for i, result in enumerate(all_results):
            result["rank"] = i + 1
        
        # Re-evaluate best parameters to get CV scores
        model = self._clone_estimator(estimator)
        for param, value in best_params.items():
            setattr(model, param, value)
        
        cv_scores = []
        for split in cv_splits:
            split_data = self.cv.get_train_test_data(split, X, y)
            X_train, X_test = split_data["X_train"], split_data["X_test"]
            y_train, y_test = split_data["y_train"], split_data["y_test"]
            
            model.fit(X_train, y_train)
            score = self._compute_score(model, X_test, y_test)
            cv_scores.append(score)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,  # Will be updated later
            method=OptimizationMethod.TPE,
            model_type=estimator.__class__.__name__,
            cv_scores=cv_scores,
            metadata={"study": study}
        )
    
    def _evolutionary_optimization(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform evolutionary optimization.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        # Get CV splits
        cv_splits = self.cv.split(X, y)
        
        # Define parameter space
        param_space = {}
        for param_name, param_info in params_dict.items():
            if param_info["type"] in ["categorical", "boolean"]:
                param_space[param_name] = param_info["values"]
            elif param_info["type"] in ["float", "int"]:
                range_min, range_max = param_info["range"]
                param_space[param_name] = (range_min, range_max, param_info["type"], param_info["log_scale"])
        
        # Define fitness function
        def fitness_function(individual):
            # Convert individual to parameters
            params = {}
            for param_name, gene in zip(param_space.keys(), individual):
                if isinstance(param_space[param_name], list):
                    # Categorical parameter
                    params[param_name] = param_space[param_name][gene]
                else:
                    # Numerical parameter
                    range_min, range_max, param_type, log_scale = param_space[param_name]
                    if log_scale:
                        if range_min <= 0:
                            range_min = 1e-6 if param_type == "float" else 1
                        # Gene is in [0, 1], convert to log scale
                        log_min = np.log(range_min)
                        log_max = np.log(range_max)
                        value = np.exp(log_min + gene * (log_max - log_min))
                    else:
                        # Linear scale
                        value = range_min + gene * (range_max - range_min)
                    
                    if param_type == "int":
                        value = int(round(value))
                    
                    params[param_name] = value
            
            # Clone estimator
            model = self._clone_estimator(estimator)
            
            # Set parameters
            for param, value in params.items():
                setattr(model, param, value)
            
            # Evaluate on CV splits
            scores = []
            for split in cv_splits:
                # Get train/test data
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Score model
                score = self._compute_score(model, X_test, y_test)
                scores.append(score)
            
            # Return mean score across splits
            return np.mean(scores),
        
        # Set up genetic algorithm
        try:
            import deap
            from deap import base, creator, tools, algorithms
            
            # Create types
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Initialize toolbox
            toolbox = base.Toolbox()
            
            # Register gene (parameter) generators
            for param_name, param_spec in param_space.items():
                if isinstance(param_spec, list):
                    # Categorical parameter
                    toolbox.register(
                        f"gene_{param_name}",
                        random.randint, 0, len(param_spec) - 1
                    )
                else:
                    # Numerical parameter - gene is always in [0, 1]
                    toolbox.register(
                        f"gene_{param_name}",
                        random.random
                    )
            
            # Register individual (chromosome) generator
            gene_generators = [getattr(toolbox, f"gene_{param_name}") for param_name in param_space.keys()]
            toolbox.register(
                "individual",
                tools.initCycle,
                creator.Individual,
                tuple(gene_generators),
                n=1
            )
            
            # Register population generator
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Register genetic operators
            toolbox.register("evaluate", fitness_function)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Initialize population
            population_size = min(50, self.n_iterations)
            population = toolbox.population(n=population_size)
            
            # Track hall of fame
            hof = tools.HallOfFame(1)
            
            # Track statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run evolution
            n_generations = max(1, self.n_iterations // population_size)
            
            if self.verbose > 0:
                logger.info(f"Running evolutionary optimization with {population_size} individuals for {n_generations} generations")
            
            population, logbook = algorithms.eaSimple(
                population, toolbox,
                cxpb=0.5,  # Crossover probability
                mutpb=0.2,  # Mutation probability
                ngen=n_generations,
                stats=stats,
                halloffame=hof,
                verbose=self.verbose > 0
            )
            
            # Get best individual
            best_individual = hof[0]
            best_fitness = best_individual.fitness.values[0]
            
            # Convert to parameters
            best_params = {}
            for param_name, gene in zip(param_space.keys(), best_individual):
                if isinstance(param_space[param_name], list):
                    # Categorical parameter
                    best_params[param_name] = param_space[param_name][gene]
                else:
                    # Numerical parameter
                    range_min, range_max, param_type, log_scale = param_space[param_name]
                    if log_scale:
                        if range_min <= 0:
                            range_min = 1e-6 if param_type == "float" else 1
                        # Gene is in [0, 1], convert to log scale
                        log_min = np.log(range_min)
                        log_max = np.log(range_max)
                        value = np.exp(log_min + gene * (log_max - log_min))
                    else:
                        # Linear scale
                        value = range_min + gene * (range_max - range_min)
                    
                    if param_type == "int":
                        value = int(round(value))
                    
                    best_params[param_name] = value
            
            # Get all results
            all_results = []
            for i, ind in enumerate(sorted(population, key=lambda x: x.fitness.values[0], reverse=True)):
                # Convert to parameters
                params = {}
                for param_name, gene in zip(param_space.keys(), ind):
                    if isinstance(param_space[param_name], list):
                        params[param_name] = param_space[param_name][gene]
                    else:
                        range_min, range_max, param_type, log_scale = param_space[param_name]
                        if log_scale:
                            if range_min <= 0:
                                range_min = 1e-6 if param_type == "float" else 1
                            log_min = np.log(range_min)
                            log_max = np.log(range_max)
                            value = np.exp(log_min + gene * (log_max - log_min))
                        else:
                            value = range_min + gene * (range_max - range_min)
                        
                        if param_type == "int":
                            value = int(round(value))
                        
                        params[param_name] = value
                
                result = {
                    "params": params,
                    "mean_test_score": ind.fitness.values[0],
                    "std_test_score": 0.0,  # Not tracked in evolution
                    "rank": i + 1
                }
                all_results.append(result)
            
            # Re-evaluate best parameters to get CV scores
            model = self._clone_estimator(estimator)
            for param, value in best_params.items():
                setattr(model, param, value)
            
            cv_scores = []
            for split in cv_splits:
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                model.fit(X_train, y_train)
                score = self._compute_score(model, X_test, y_test)
                cv_scores.append(score)
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_fitness,
                all_results=all_results,
                optimization_time=0,  # Will be updated later
                method=OptimizationMethod.EVOLUTIONARY,
                model_type=estimator.__class__.__name__,
                cv_scores=cv_scores,
                metadata={"logbook": logbook}
            )
        
        except ImportError:
            logger.warning("DEAP not available. Using random search instead.")
            return self._random_search(estimator, params_dict, X, y)
        
        finally:
            # Clean up DEAP global variables to avoid conflicts with future runs
            if 'creator' in globals():
                if hasattr(creator, 'FitnessMax'):
                    delattr(creator, 'FitnessMax')
                if hasattr(creator, 'Individual'):
                    delattr(creator, 'Individual')
    
    def _hyperband_optimization(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform Hyperband optimization.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("optuna not available. Using random search instead.")
            return self._random_search(estimator, params_dict, X, y)
        
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import HyperbandPruner
        
        # This is similar to TPE but with early stopping
        
        # Get CV splits
        cv_splits = self.cv.split(X, y)
        
        # Define the objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_info in params_dict.items():
                if param_info["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_info["values"])
                elif param_info["type"] == "boolean":
                    params[param_name] = trial.suggest_categorical(param_name, [True, False])
                elif param_info["type"] == "float":
                    range_min, range_max = param_info["range"]
                    if param_info["log_scale"]:
                        if range_min <= 0:
                            range_min = 1e-6
                        params[param_name] = trial.suggest_float(param_name, range_min, range_max, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, range_min, range_max)
                elif param_info["type"] == "int":
                    range_min, range_max = param_info["range"]
                    if param_info["log_scale"]:
                        if range_min <= 0:
                            range_min = 1
                        params[param_name] = trial.suggest_int(param_name, range_min, range_max, log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, range_min, range_max)
            
            # Clone estimator
            model = self._clone_estimator(estimator)
            
            # Set parameters
            for param, value in params.items():
                setattr(model, param, value)
            
            # Evaluate on CV splits with early stopping
            scores = []
            for i, split in enumerate(cv_splits):
                # Report intermediate result for pruning
                trial.report(np.mean(scores) if scores else 0.0, i)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # Get train/test data
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Score model
                score = self._compute_score(model, X_test, y_test)
                scores.append(score)
            
            # Return mean score across splits
            return np.mean(scores)
        
        # Create Optuna study with Hyperband pruner
        sampler = TPESampler(seed=self.random_state)
        pruner = HyperbandPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_iterations,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose > 0
        )
        
        # Get best parameters and score
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Format all results
        all_results = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    "params": trial.params,
                    "mean_test_score": trial.value,
                    "std_test_score": 0.0,  # Not available from Optuna
                    "rank": i + 1
                }
                all_results.append(result)
        
        # Sort by mean test score (descending)
        all_results = sorted(all_results, key=lambda x: x["mean_test_score"], reverse=True)
        
        # Add rank to results
        for i, result in enumerate(all_results):
            result["rank"] = i + 1
        
        # Re-evaluate best parameters to get CV scores
        model = self._clone_estimator(estimator)
        for param, value in best_params.items():
            setattr(model, param, value)
        
        cv_scores = []
        for split in cv_splits:
            split_data = self.cv.get_train_test_data(split, X, y)
            X_train, X_test = split_data["X_train"], split_data["X_test"]
            y_train, y_test = split_data["y_train"], split_data["y_test"]
            
            model.fit(X_train, y_train)
            score = self._compute_score(model, X_test, y_test)
            cv_scores.append(score)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,  # Will be updated later
            method=OptimizationMethod.HYPERBAND,
            model_type=estimator.__class__.__name__,
            cv_scores=cv_scores,
            metadata={"study": study}
        )
    
    def _distributed_optimization(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform distributed optimization using Ray Tune.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        if not RAY_AVAILABLE:
            logger.warning("ray not available. Using random search instead.")
            return self._random_search(estimator, params_dict, X, y)
        
        from ray import tune
        from ray.tune.schedulers import HyperBandScheduler
        from ray.tune.search.optuna import OptunaSearch
        
        # Get CV splits
        cv_splits = self.cv.split(X, y)
        
        # Convert parameters to Ray Tune format
        search_space = {}
        for param_name, param_info in params_dict.items():
            if param_info["type"] == "categorical":
                search_space[param_name] = tune.choice(param_info["values"])
            elif param_info["type"] == "boolean":
                search_space[param_name] = tune.choice([True, False])
            elif param_info["type"] == "float":
                range_min, range_max = param_info["range"]
                if param_info["log_scale"]:
                    if range_min <= 0:
                        range_min = 1e-6
                    search_space[param_name] = tune.loguniform(range_min, range_max)
                else:
                    search_space[param_name] = tune.uniform(range_min, range_max)
            elif param_info["type"] == "int":
                range_min, range_max = param_info["range"]
                if param_info["log_scale"]:
                    if range_min <= 0:
                        range_min = 1
                    search_space[param_name] = tune.lograndint(range_min, range_max)
                else:
                    search_space[param_name] = tune.randint(range_min, range_max + 1)
        
        # Define the training function
        def train_model(config):
            # Clone estimator
            model = self._clone_estimator(estimator)
            
            # Set parameters
            for param, value in config.items():
                setattr(model, param, value)
            
            # Evaluate on CV splits
            scores = []
            for i, split in enumerate(cv_splits):
                # Get train/test data
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Score model
                score = self._compute_score(model, X_test, y_test)
                scores.append(score)
                
                # Report intermediate result
                tune.report(mean_score=np.mean(scores), split=i, score=score)
            
            # Report final result
            tune.report(mean_score=np.mean(scores), std_score=np.std(scores))
        
        # Configure search algorithm
        search_algo = OptunaSearch(metric="mean_score", mode="max")
        
        # Configure scheduler
        scheduler = HyperBandScheduler(
            time_attr="training_iteration",
            metric="mean_score",
            mode="max"
        )
        
        # Run Ray Tune
        analysis = tune.run(
            train_model,
            config=search_space,
            num_samples=self.n_iterations,
            scheduler=scheduler,
            search_alg=search_algo,
            verbose=self.verbose,
            resources_per_trial={"cpu": 1, "gpu": 0},
            max_concurrent_trials=self.n_jobs,
            stop={"training_iteration": len(cv_splits)}
        )
        
        # Get best parameters
        best_trial = analysis.get_best_trial("mean_score", "max", "last")
        best_params = best_trial.config
        best_score = best_trial.last_result["mean_score"]
        
        # Format all results
        all_results = []
        for i, trial in enumerate(analysis.trials):
            if trial.last_result and "mean_score" in trial.last_result:
                result = {
                    "params": trial.config,
                    "mean_test_score": trial.last_result["mean_score"],
                    "std_test_score": trial.last_result.get("std_score", 0.0),
                    "rank": i + 1
                }
                all_results.append(result)
        
        # Sort by mean test score (descending)
        all_results = sorted(all_results, key=lambda x: x["mean_test_score"], reverse=True)
        
        # Add rank to results
        for i, result in enumerate(all_results):
            result["rank"] = i + 1
        
        # Get CV scores
        cv_scores = []
        if "cv_scores" in best_trial.last_result:
            cv_scores = best_trial.last_result["cv_scores"]
        else:
            # Re-evaluate best parameters to get CV scores
            model = self._clone_estimator(estimator)
            for param, value in best_params.items():
                setattr(model, param, value)
            
            for split in cv_splits:
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]
                
                model.fit(X_train, y_train)
                score = self._compute_score(model, X_test, y_test)
                cv_scores.append(score)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,  # Will be updated later
            method=OptimizationMethod.DISTRIBUTED,
            model_type=estimator.__class__.__name__,
            cv_scores=cv_scores,
            metadata={"analysis": analysis}
        )
    
    def _compute_score(self, model, X, y):
        """
        Compute score for a fitted estimator.
        
        Args:
            model: Fitted estimator
            X: Feature data
            y: Target data
            
        Returns:
            Score value
        """
        if callable(self.scoring):
            # Custom scoring function
            y_pred = model.predict(X)
            return self.scoring(y, y_pred)
        elif isinstance(self.scoring, str):
            # Handle common scoring metrics
            if self.scoring == 'neg_mean_squared_error':
                y_pred = model.predict(X)
                return -np.mean((y - y_pred) ** 2)
            elif self.scoring == 'neg_mean_absolute_error':
                y_pred = model.predict(X)
                return -np.mean(np.abs(y - y_pred))
            elif self.scoring == 'r2':
                y_pred = model.predict(X)
                u = ((y - y_pred) ** 2).sum()
                v = ((y - y.mean()) ** 2).sum()
                return 1 - (u / v) if v > 0 else 0
            elif self.scoring == 'accuracy':
                y_pred = model.predict(X)
                return np.mean(y == y_pred)
            elif self.scoring == 'sharpe':
                # Trading-specific metric: Sharpe ratio of returns
                y_pred = model.predict(X)
                returns = y_pred * y  # Use true returns * predicted direction
                return returns.mean() / (returns.std() + 1e-6)
            elif self.scoring == 'calmar':
                # Trading-specific metric: Calmar ratio (return / max drawdown)
                y_pred = model.predict(X)
                returns = y_pred * y  # Use true returns * predicted direction
                cumulative_returns = (1 + returns).cumprod()
                max_drawdown = 1 - (cumulative_returns / cumulative_returns.cummax()).min()
                return returns.mean() / (max_drawdown + 1e-6)
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring}")
        else:
            raise ValueError("scoring should be a callable or a string")
    
    def _clone_estimator(self, estimator):
        """Clone the estimator"""
        try:
            from sklearn.base import clone
            return clone(estimator)
        except ImportError:
            # If sklearn is not available, try a simple copy
            import copy
            return copy.deepcopy(estimator)


# Factory function for creating optimizers
def create_optimizer(method: str = "bayesian", **kwargs) -> HyperparameterOptimizer:
    """
    Create a hyperparameter optimizer.
    
    Args:
        method: Optimization method name
        **kwargs: Additional parameters for the optimizer
        
    Returns:
        HyperparameterOptimizer instance
    """
    try:
        opt_method = OptimizationMethod(method)
    except ValueError:
        raise ValueError(f"Unknown optimization method: {method}. "
                         f"Available methods: {[m.value for m in OptimizationMethod]}")
    
    return HyperparameterOptimizer(method=opt_method, **kwargs)


# Factory function for creating parameters
def create_parameter(name: str, 
                    type: str, 
                    values: Union[List[Any], Tuple[Any, Any]],
                    log_scale: bool = False,
                    step: Optional[Union[int, float]] = None,
                    depends_on: Optional[Dict[str, Any]] = None) -> Parameter:
    """
    Create a parameter for optimization.
    
    Args:
        name: Parameter name
        type: Parameter type ('float', 'int', 'categorical', 'boolean')
        values: List of values for categorical or range (min, max) for numerical
        log_scale: Whether to use log scale for numerical parameters
        step: Step size for numerical parameters
        depends_on: Conditional dependencies
    
    Returns:
        Parameter instance
    """
    return Parameter(
        name=name,
        type=type,
        values=values,
        log_scale=log_scale,
        step=step,
        depends_on=depends_on
    )"""
hyperparameter_optimization.py - Hyperparameter Optimization Framework

This module provides hyperparameter optimization utilities for machine learning models
in the trading system, including grid search, random search, Bayesian optimization,
and evolutionary algorithms.
"""

import logging
import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Iterator, Set
from enum import Enum
from dataclasses import dataclass, field
import random
import threading
import concurrent.futures
import warnings

# Import custom modules
from models.research.cross_validation import TimeSeriesCV, create_time_series_cv

# Try to import optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Methods for hyperparameter optimization"""
    GRID_SEARCH = "grid_search"              # Grid search
    RANDOM_SEARCH = "random_search"          # Random search
    BAYESIAN_OPTIMIZATION = "bayesian"       # Bayesian optimization
    EVOLUTIONARY = "evolutionary"            # Evolutionary algorithms
    TPE = "tpe"                              # Tree-structured Parzen Estimator
    HYPERBAND = "hyperband"                  # Hyperband optimization
    DISTRIBUTED = "distributed"              # Distributed optimization


@dataclass
class Parameter:
    """Parameter definition for optimization"""
    name: str
    type: str  # 'float', 'int', 'categorical', 'boolean'
    values: Union[List[Any], Tuple[Any, Any]] = None  # List for categorical, tuple for range
    log_scale: bool = False  # Whether to use log scale for numerical parameters
    step: Optional[Union[int, float]] = None  # Step size for numerical parameters
    depends_on: Optional[Dict[str, Any]] = None  # Conditional dependencies


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    method: OptimizationMethod
    model_type: str
    cv_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization framework with support for multiple methods
    and time series cross-validation.
    """
    
    def __init__(self,
                 method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION,
                 cv: Optional[TimeSeriesCV] = None,
                 scoring: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'neg_mean_squared_error',
                 n_jobs: int = 1,
                 n_iterations: int = 100,
                 random_state: Optional[int] = None,
                 timeout: Optional[int] = None,
                 verbose: int = 0,
                 refit: bool = True,
                 early_stopping: bool = True,
                 progress_callback: Optional[Callable[[float, Dict[str, Any], Dict[str, Any]], None]] = None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            method: Optimization method
            cv: Time series cross-validation strategy
            scoring: Scoring method or function(y_true, y_pred) -> float
            n_jobs: Number of parallel jobs
            n_iterations: Number of iterations (for random search, bayesian, etc.)
            random_state: Random seed
            timeout: Maximum optimization time in seconds
            verbose: Verbosity level
            refit: Whether to refit the model on the best parameters
            early_stopping: Whether to use early stopping
            progress_callback: Optional callback function for progress updates
        """
        self.method = method
        self.cv = cv or create_time_series_cv()
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.timeout = timeout
        self.verbose = verbose
        self.refit = refit
        self.early_stopping = early_stopping
        self.progress_callback = progress_callback
        
        # Check if the selected method is available
        if method == OptimizationMethod.BAYESIAN_OPTIMIZATION and not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available. Falling back to random search.")
            self.method = OptimizationMethod.RANDOM_SEARCH
        
        if method == OptimizationMethod.TPE and not OPTUNA_AVAILABLE:
            logger.warning("optuna not available. Falling back to random search.")
            self.method = OptimizationMethod.RANDOM_SEARCH
        
        if method == OptimizationMethod.DISTRIBUTED and not RAY_AVAILABLE:
            logger.warning("ray not available. Falling back to random search.")
            self.method = OptimizationMethod.RANDOM_SEARCH
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        
        logger.info(f"HyperparameterOptimizer initialized with method {method.value}")
    
    def optimize(self,
                estimator: Any,
                parameters: List[Parameter],
                X: Union[pd.DataFrame, np.ndarray],
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                model_type: str = "model") -> OptimizationResult:
        """
        Optimize hyperparameters for a model.
        
        Args:
            estimator: Model to optimize
            parameters: List of parameter definitions
            X: Feature data
            y: Target data
            model_type: Type of model (for logging)
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        
        # Convert parameters to the format expected by the optimization method
        params_dict = self._format_parameters(parameters)
        
        if self.verbose > 0:
            param_info = ", ".join([f"{p.name}({p.type})" for p in parameters])
            logger.info(f"Starting hyperparameter optimization for {model_type} with {len(parameters)} parameters: {param_info}")
            logger.info(f"Method: {self.method.value}, iterations: {self.n_iterations}")
        
        # Run the appropriate optimization method
        if self.method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search(estimator, params_dict, X, y)
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search(estimator, params_dict, X, y)
        elif self.method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
            result = self._bayesian_optimization(estimator, params_dict, X, y)
        elif self.method == OptimizationMethod.EVOLUTIONARY:
            result = self._evolutionary_optimization(estimator, params_dict, X, y)
        elif self.method == OptimizationMethod.TPE:
            result = self._tpe_optimization(estimator, params_dict, X, y)
        elif self.method == OptimizationMethod.HYPERBAND:
            result = self._hyperband_optimization(estimator, params_dict, X, y)
        elif self.method == OptimizationMethod.DISTRIBUTED:
            result = self._distributed_optimization(estimator, params_dict, X, y)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
        optimization_time = time.time() - start_time
        
        # Refit on the best parameters if requested
        best_estimator = None
        if self.refit:
            if self.verbose > 0:
                logger.info(f"Refitting model with best parameters: {result.best_params}")
            
            best_estimator = self._clone_estimator(estimator)
            
            # Set best parameters
            for param, value in result.best_params.items():
                setattr(best_estimator, param, value)
            
            # Fit on full dataset
            best_estimator.fit(X, y)
            
            # Add to result metadata
            result.metadata["best_estimator"] = best_estimator
        
        if self.verbose > 0:
            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            logger.info(f"Best parameters: {result.best_params}")
            logger.info(f"Best score: {result.best_score:.4f}")
        
        # Update optimization time
        result.optimization_time = optimization_time
        
        return result
    
    def _format_parameters(self, parameters: List[Parameter]) -> Dict[str, Dict[str, Any]]:
        """
        Format parameters for different optimization methods.
        
        Args:
            parameters: List of parameter definitions
            
        Returns:
            Dictionary of parameters in the format expected by the optimization method
        """
        params_dict = {}
        
        for param in parameters:
            param_info = {"type": param.type}
            
            if param.type == "categorical":
                param_info["values"] = param.values
            elif param.type in ["float", "int"]:
                param_info["range"] = param.values
                param_info["log_scale"] = param.log_scale
                if param.step is not None:
                    param_info["step"] = param.step
            elif param.type == "boolean":
                param_info["values"] = [True, False]
            
            if param.depends_on:
                param_info["depends_on"] = param.depends_on
            
            params_dict[param.name] = param_info
        
        return params_dict
    
    def _grid_search(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        from models.research.cross_validation import TimeSeriesGridSearchCV
        
        # Convert parameters to grid format
        param_grid = {}
        for param_name, param_info in params_dict.items():
            if param_info["type"] in ["categorical", "boolean"]:
                param_grid[param_name] = param_info["values"]
            elif param_info["type"] in ["float", "int"]:
                range_min, range_max = param_info["range"]
                
                if param_info["type"] == "int":
                    step = param_info.get("step", 1)
                    values = list(range(range_min, range_max + 1, step))
                else:  # float
                    # For floats, create a reasonable number of values in the range
                    step = param_info.get("step")
                    if step is not None:
                        values = np.arange(range_min, range_max + step, step).tolist()
                    else:
                        # Default to 10 values if no step provided
                        values = np.linspace(range_min, range_max, 10).tolist()
                
                param_grid[param_name] = values
        
        # Initialize grid search
        grid_search = TimeSeriesGridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=False,  # We'll refit manually if needed
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Get results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Format all results
        all_results = []
        for i, params in enumerate(grid_search.cv_results_["params"]):
            result = {
                "params": params,
                "mean_test_score": grid_search.cv_results_["mean_test_score"][i],
                "std_test_score": grid_search.cv_results_["std_test_score"][i],
                "rank": i + 1
            }
            all_results.append(result)
        
        # Sort by mean test score (descending)
        all_results = sorted(all_results, key=lambda x: x["mean_test_score"], reverse=True)
        
        # Get CV scores for best parameters
        best_idx = grid_search.best_index_
        cv_scores = grid_search.cv_results_["split_test_scores"][best_idx]
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=0,  # Will be updated later
            method=OptimizationMethod.GRID_SEARCH,
            model_type=estimator.__class__.__name__,
            cv_scores=cv_scores
        )
    
    def _random_search(self, estimator, params_dict, X, y) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Args:
            estimator: Model to optimize
            params_dict: Dictionary of parameters
            X: Feature data
            y: Target data
            
        Returns:
            Optimization result
        """
        from sklearn.model_selection import ParameterSampler
        
        # Get CV splits
        cv_splits = self.cv.split(X, y)
        
        # Convert parameters to search space
        param_distributions = {}
        for param_name, param_info in params_dict.items():
            if param_info["type"] in ["categorical", "boolean"]:
                param_distributions[param_name] = param_info["values"]
            elif param_info["type"] in ["float", "int"]:
                range_min, range_max = param_info["range"]
                
                if param_info["log_scale"]:
                    # Log-uniform distribution
                    if range_min <= 0:
                        range_min = 1e-6  # Avoid non-positive values for log scale
                    
                    if param_info["type"] == "int":
                        param_distributions[param_name] = np.logspace(
                            np.log10(range_min), np.log10(range_max), 100
                        ).astype(int)
                    else:
                        param_distributions[param_name] = np.logspace(
                            np.log10(range_min), np.log10(range_max), 100
                        )
                else:
                    # Uniform distribution
                    if param_info["type"] == "int":
                        param_distributions[param_name] = range(range_min, range_max + 1)
                    else:
                        param_distributions[param_name] = np.linspace(range_min, range_max, 100)
        
        # Generate random parameter combinations
        param_sampler = ParameterSampler(
            param_distributions, 
            n_iter=self.n_iterations,
            random_state=self.random_state
        )
        
        # Perform random search
        all_results = []
        best_score = float('-inf')
        best_params = None
        
        # Use progress bar if available
        try:
            from tqdm import tqdm
            param_list = list(param_sampler)
            iterator = tqdm(param_list) if self.verbose > 0 else param_list
        except ImportError:
            iterator = param_sampler
        
        # Prepare function for parallel execution
        def evaluate_params(params):
            # Clone estimator
            model = self._clone