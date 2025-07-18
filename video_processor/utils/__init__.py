# Utils package for video processor
# Import all utility functions from the main utils.py file to maintain compatibility

import os
import importlib.util

# Direct import from the utils.py file to avoid naming conflicts
utils_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
spec = importlib.util.spec_from_file_location("video_processor_utils", utils_file_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

# Import all the utility functions
timeout = utils_module.timeout
generate_idempotency_key = utils_module.generate_idempotency_key
check_task_idempotency = utils_module.check_task_idempotency
mark_task_complete = utils_module.mark_task_complete
idempotent_task = utils_module.idempotent_task
handle_dead_letter_task = utils_module.handle_dead_letter_task
atomic_with_callback = utils_module.atomic_with_callback
update_task_progress = utils_module.update_task_progress

# Re-export all imported functions
__all__ = [
    'timeout',
    'generate_idempotency_key', 
    'check_task_idempotency',
    'mark_task_complete',
    'idempotent_task',
    'handle_dead_letter_task',
    'atomic_with_callback',
    'update_task_progress'
] 