import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class ParallelTaskRunner:
    def __init__(self, task_function, task_configs, max_workers, main_logger_name):
        """
        Manages parallel execution of a given task function using ThreadPoolExecutor.

        Args:
            task_function: The function to execute for each task.
                           It should accept a single argument (a dictionary or object)
                           containing all necessary configuration for that task.
            task_configs: A list of configuration objects/dictionaries, one for each task.
            max_workers: Maximum number of worker threads.
            main_logger_name: Name of the main logger to use for the runner's own logging.
        """
        self.task_function = task_function
        self.task_configs = task_configs
        self.max_workers = max_workers
        self.logger = logging.getLogger(main_logger_name)
        self.results = []

    def run(self):
        self.logger.info(f"Starting parallel execution with up to {self.max_workers} workers for {len(self.task_configs)} tasks.")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_config = {
                executor.submit(self.task_function, config): config
                for config in self.task_configs
            }

            for future in as_completed(future_to_config):
                task_config_completed = future_to_config[future]
                try:
                    result = future.result() # This will re-raise exceptions from the task_function
                    self.results.append(result)
                except Exception as exc:
                    self.logger.error(f"Task with config (first 100 chars): '{str(task_config_completed)[:100]}' generated an exception: {exc}", exc_info=True)
                    # The task_function itself should return a dict with status='error'
                    # This catch is for unexpected errors in the future.result() or executor itself.
                    self.results.append({'task_config': str(task_config_completed)[:100], 'status': 'runner_exception', 'error_message': str(exc)})
        
        self.logger.info(f"Parallel execution finished. Collected {len(self.results)} results/statuses.")
        return self.results