
import asyncio
import importlib.util
import logging
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configure logger to output to stderr with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

SEED = 42

def chat_completion_request_openai(prompt, client):
    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_response = client.chat.completions.create(
    model='gpt-5-mini',
    messages=messages)
    if chat_response.choices:
        completion_text = chat_response.choices[0].message.content
    else:
        completion_text = None
    return completion_text

definitions = open(Path(Path(__file__).parent / "resources" / "taxonomy_definitions_examples" / "definitions.txt"), "r").read()
examples = open(Path(Path(__file__).parent / "resources" / "taxonomy_definitions_examples" / "examples.txt"), "r").read()

def openai_evaluator(trace, definitions=definitions, examples=examples, client=None):
    prompt = (
    "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
    "In the traces, analyze the system behaviour."
    "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
    "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
    "Also tell me whether the task is successfully completed or not, as a binary yes or no."
    "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
    "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
    "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
    "*** begin of things you should answer *** @@"
    "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
    "B. Whether the task is successfully completed or not: <yes or no>"
    "C. Whether you encounter any of the failure modes or inefficiencies:"
    "1.1 Disobey Task Specification: <yes or no>"
    "1.2 Disobey Role Specification: <yes or no>"
    "1.3 Step Repetition: <yes or no>"
    "1.4 Loss of Conversation History: <yes or no>"
    "1.5 Unaware of Termination Conditions: <yes or no>"
    "2.1 Conversation Reset: <yes or no>"
    "2.2 Fail to Ask for Clarification: <yes or no>"
    "2.3 Task Derailment: <yes or no>"
    "2.4 Information Withholding: <yes or no>"
    "2.5 Ignored Other Agent's Input: <yes or no>"
    "2.6 Action-Reasoning Mismatch: <yes or no>"
    "3.1 Premature Termination: <yes or no>"
    "3.2 No or Incorrect Verification: <yes or no>"
    "3.3 Weak Verification: <yes or no>"
    "@@*** end of your answer ***"
    "An example answer is: \n"
    "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
    "B. no \n"
    "C. \n"
    "1.1 no \n"
    "1.2 no \n"
    "1.3 no \n"
    "1.4 no \n"
    "1.5 no \n"
    "1.6 yes \n"
    "2.1 no \n"
    "2.2 no \n"
    "2.3 yes \n"
    "2.4 no \n"
    "2.5 no \n"
    "2.6 yes \n"
    "2.7 no \n"
    "3.1 no \n"
    "3.2 yes \n"
    "3.3 no \n"   
    "Here is the trace: \n"
    f"{trace}"
    "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
    f"{definitions} \n"
    "Here are some examples of the failure modes and inefficiencies: \n"
    f"{examples}"
)
    return chat_completion_request_openai(prompt, client)

class MASTLLMJudge:
    def __init__(self, api_key: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("OpenAI package is required but not installed")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)

    def evaluate_trace(self, trace: str) -> Dict:
        try:
            raw_response = openai_evaluator(trace, client=self.client)

            failure_modes = self._parse_response_list([raw_response])
            
            return {
                "failure_modes": failure_modes,
                "total_failures": sum(failure_modes.values()),
                "raw_response": raw_response
            }
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            failure_modes = {f'{i}.{j}': 0 for i in range(1,4) for j in range(1,7) if not (i==1 and j>5) and not (i==3 and j>3)}
            return {
                "failure_modes": failure_modes,
                "total_failures": 0,
                "raw_response": str(e)
            }

    def _call_llm(self, prompt: str) -> str:
        return self.client.chat.completions.create(
            model='o1',
            messages=[
                {"role": "system", "content": "You are an expert at analyzing multi-agent system failures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
    
    def _parse_response(self, response: str) -> Dict[str, int]:
        failure_modes = {
            '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0,
            '2.1': 0, '2.2': 0, '2.3': 0, '2.4': 0, '2.5': 0, '2.6': 0,
            '3.1': 0, '3.2': 0, '3.3': 0
        }
        
        for mode in failure_modes.keys():
            pattern = rf"{mode}.*?(yes|no)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                failure_modes[mode] = 1 if match.group(1).lower() == 'yes' else 0
        
        return failure_modes

    def _parse_response_list(self, responses: List[str]) -> Dict[str, int]:
        import re
        
        # Initialize dictionary with empty lists for each failure mode
        failure_modes = {
            '1.1': [], '1.2': [], '1.3': [], '1.4': [], '1.5': [],
            '2.1': [], '2.2': [], '2.3': [], '2.4': [], '2.5': [], '2.6': [],
            '3.1': [], '3.2': [], '3.3': []
        }
        
        for i, response in enumerate(responses):
            try:
                # Clean up the response - remove @@ markers if present
                cleaned_response = response.strip()
                if cleaned_response.startswith('@@'):
                    cleaned_response = cleaned_response[2:]
                if cleaned_response.endswith('@@'):
                    cleaned_response = cleaned_response[:-2]
                
                # Process each failure mode
                for mode in failure_modes.keys():
                    # Various patterns to match different response formats
                    patterns = [
                        # Format with C. prefix and colon
                        rf"C\..*?{mode}.*?(yes|no)",
                        # Format with just C prefix without dot
                        rf"C{mode}\s+(yes|no)",
                        # Format with mode directly (with or without spaces)
                        rf"{mode}\s*[:]\s*(yes|no)",
                        rf"{mode}\s+(yes|no)",
                        # Format with newlines
                        rf"{mode}\s*\n\s*(yes|no)",
                        # Format with C prefix and newlines
                        rf"C\.{mode}\s*\n\s*(yes|no)"
                    ]
                    
                    found = False
                    for pattern in patterns:
                        matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                        if matches:
                            # Use the first match
                            value = 1 if matches[0].lower() == 'yes' else 0
                            failure_modes[mode].append(value)
                            found = True
                            break
                    
                    if not found:
                        # If we still can't find a match, try a more general approach
                        # Look for the mode number followed by any text and then yes/no
                        general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                        match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                        
                        if match:
                            value = 1 if match.group(1).lower() == 'yes' else 0
                            failure_modes[mode].append(value)
                        else:
                            # If all attempts fail, default to 'no'
                            print(f"Warning: Could not find mode {mode} in response {i}")
                            failure_modes[mode].append(0)
                        
            except Exception as e:
                print(f"Error processing response {i}: {e}")
                # If there's an error, default to 'no' for all modes for this response
                for mode in failure_modes:
                    if len(failure_modes[mode]) <= i:  # Only append if we haven't already
                        failure_modes[mode].append(0)
        
        # Ensure all lists have the same length
        max_length = max(len(values) for values in failure_modes.values())
        for mode in failure_modes:
            if len(failure_modes[mode]) < max_length:
                failure_modes[mode].extend([0] * (max_length - len(failure_modes[mode])))
        
        failure_modes_numeric = {
            '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0,
            '2.1': 0, '2.2': 0, '2.3': 0, '2.4': 0, '2.5': 0, '2.6': 0,
            '3.1': 0, '3.2': 0, '3.3': 0
        }

        if len(responses) == 1:
            for mode in failure_modes_numeric:
                failure_modes_numeric[mode] = sum(failure_modes[mode])
        else:
            for mode in failure_modes_numeric:
                failure_modes_numeric[mode] = sum(failure_modes[mode]) / len(failure_modes[mode])

        return failure_modes_numeric

class ProgramDevDataset:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            env_override = os.getenv("MAST_DATA_DIR")
            if env_override:
                data_dir = env_override
            else:
                problem_dir = Path(__file__).parent
                # First try mounted location (for containers)
                mounted_datasets = Path("/datasets/multiagent_system/openevolve-mast/example_mas/programdev")
                if mounted_datasets.exists() and any(mounted_datasets.iterdir()):
                    data_dir = str(mounted_datasets)
                else:
                    # Then try local resources/datasets (for local execution)
                    local_datasets = problem_dir / "resources" / "datasets"
                    if local_datasets.exists() and any(local_datasets.iterdir()):
                        data_dir = str(local_datasets)
                    else:
                        # Fallback to resources/openevolve-mast/example_mas/programdev
                        fallback_datasets = problem_dir / "resources" / "openevolve-mast" / "example_mas" / "programdev"
                        if fallback_datasets.exists() and any(fallback_datasets.iterdir()):
                            data_dir = str(fallback_datasets)
                        else:
                            # Last fallback: old location for backward compatibility
                            base = problem_dir.parent.parent  # problems/multiagent_system -> repo root
                            datasets_root = base / "datasets" / "multiagent_system" / "openevolve-mast" / "example_mas" / "programdev"
                            data_dir = str(datasets_root)
        self.data_dir = Path(data_dir)
        self.tasks = []
        self._load_tasks()
    
    def _load_tasks(self):
        name_files = sorted(self.data_dir.glob("names_*.txt"))
        
        for name_file in name_files:
            index = name_file.stem.split("_")[1]
            desc_file = self.data_dir / f"descriptions_{index}.txt"
            
            if desc_file.exists():
                with open(name_file, 'r') as f:
                    name = f.read().strip()
                with open(desc_file, 'r') as f:
                    description = f.read().strip()
                
                if name and description:
                    self.tasks.append({
                        'index': index,
                        'name': name,
                        'description': description
                    })
        
        logger.info(f"Loaded {len(self.tasks)} tasks from {self.data_dir}")
    
    def sample_tasks(self, n: int = 3) -> List[Dict]:
        if n > len(self.tasks):
            n = len(self.tasks)
        return random.sample(self.tasks, n)


def evaluate(program_path: str) -> Dict:
    try:
        # Ensure program_path is absolute
        program_path = str(Path(program_path).resolve())
        
        # Check if file exists
        if not os.path.exists(program_path):
            return {'score': 0.0, 'runs_successfully': 0.0, 'error': f'Program file not found: {program_path}'}
        
        # Load the program module
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            return {'score': 0.0, 'runs_successfully': 0.0, 'error': f'Cannot load program module from: {program_path}'}
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if required function exists
        if not hasattr(program, 'run_multi_agent_task'):
            return {'score': 0.0, 'runs_successfully': 0.0, 'error': 'Missing run_multi_agent_task function'}
        
        # Initialize components
        logger.info("Initializing dataset...")
        dataset = ProgramDevDataset()
        logger.info(f"Dataset initialized. Loaded {len(dataset.tasks)} tasks")
        
        logger.info("Initializing LLM judge...")
        judge = MASTLLMJudge()
        logger.info("LLM judge initialized")
        
        logger.info("Sampling tasks for evaluation...")
        # Set fixed seed for reproducible task sampling across all evaluations
        random.seed(SEED)
        sample_tasks = dataset.sample_tasks(6)  # Evaluate on 6 random tasks
        logger.info(f"Sampled {len(sample_tasks)} tasks for evaluation")
        
        if not sample_tasks:
            return {'score': 0.0, 'runs_successfully': 0.0, 'error': 'No tasks in dataset'}
        
        # Run evaluation on each task
        task_scores = []  # Store normalized scores for each task
        task_failures = []  # Store failures for each task
        total_failures = 0
        successful_runs = 0
        
        # Score normalization parameters
        min_raw_score = 1.0 / 15.0  # 1/(1+14) - worst case
        max_raw_score = 1.0  # 1/(1+0) - best case
        
        logger.info(f"Starting evaluation on {len(sample_tasks)} tasks...")
        for idx, task in enumerate(sample_tasks, 1):
            logger.info(f"[Task {idx}/{len(sample_tasks)}] Evaluating task: {task.get('name', 'unknown')}")
            task_failures_count = None
            try:
                logger.info(f"[Task {idx}/{len(sample_tasks)}] Creating temporary log file...")
                # Create temporary log file for the trace
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as tmp_file:
                    log_file = tmp_file.name
                logger.info(f"[Task {idx}/{len(sample_tasks)}] Log file: {log_file}")
                
                logger.info(f"[Task {idx}/{len(sample_tasks)}] Running multi-agent task (timeout: 60s)...")
                # Run the multi-agent task with timeout
                async def run_with_timeout():
                    try:
                        return await asyncio.wait_for(
                            program.run_multi_agent_task(
                                idea=task['description'],
                                log_file=log_file
                            ),
                            timeout=60
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"[Task {idx}/{len(sample_tasks)}] Task timed out after 60 seconds")
                        return None
                
                trace = asyncio.run(run_with_timeout())
                
                if trace is None:
                    logger.warning(f"[Task {idx}/{len(sample_tasks)}] Task failed (timeout), adding penalty")
                    task_failures_count = 7  # Penalize timeout
                    total_failures += task_failures_count
                else:
                    logger.info(f"[Task {idx}/{len(sample_tasks)}] Task completed. Trace length: {len(trace) if trace else 0} chars")
                    logger.info(f"[Task {idx}/{len(sample_tasks)}] Evaluating trace with LLM judge...")
                    # Evaluate the trace with LLM judge
                    evaluation = judge.evaluate_trace(trace)
                    task_failures_count = evaluation['total_failures']
                    total_failures += task_failures_count
                    successful_runs += 1
                    
                    logger.info(f"[Task {idx}/{len(sample_tasks)}] Task {task['name']}: {task_failures_count} failures, total so far: {total_failures}")
                
                # Clean up temp file
                try:
                    os.unlink(log_file)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Error evaluating task {task['name']}: {str(e)}")
                task_failures_count = 14
                total_failures += task_failures_count
            
            # Calculate score for this task
            if task_failures_count is not None:
                # Clamp failures to valid range [0, 14]
                task_failures_count = max(0.0, min(14.0, float(task_failures_count)))
                task_failures.append(task_failures_count)
                
                # Calculate raw score: 1 / (1 + failures)
                raw_score = 1.0 / (1.0 + task_failures_count)
                
                # Normalize to 0-100%: (raw_score - min) / (max - min) * 100
                normalized_score = ((raw_score - min_raw_score) / (max_raw_score - min_raw_score)) * 100.0
                normalized_score = max(0.0, min(100.0, normalized_score))
                task_scores.append(normalized_score)
        
        # Calculate average normalized score across all tasks
        if len(task_scores) > 0:
            avg_normalized_score = sum(task_scores) / len(task_scores)
            avg_failures_per_task = sum(task_failures) / len(task_failures) if task_failures else 14.0
        else:
            avg_normalized_score = 0.0
            avg_failures_per_task = 14.0

        return {
            'score': float(avg_normalized_score),
            'runs_successfully': 1.0 if successful_runs > 0 else 0.0,
            'avg_failures_per_task': float(avg_failures_per_task),
            'raw_score': float(1.0 / (1.0 + avg_failures_per_task)),
            'total_failures': int(total_failures),
            'successful_runs': int(successful_runs)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {'score': 0.0, 'runs_successfully': 0.0, 'error': str(e)}


def evaluate_stage1(program_path: str) -> Dict:
    try:
        # Ensure program_path is absolute
        program_path = str(Path(program_path).resolve())
        
        # Check if file exists
        if not os.path.exists(program_path):
            return {
                'runs_successfully': 0.0, 
                'score': 0.0,
                'avg_failures_per_task': 14.0,
                'error': f'Program file not found: {program_path}'
            }
        
        # Load the program module
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            return {
                'runs_successfully': 0.0, 
                'score': 0.0,
                'avg_failures_per_task': 14.0,
                'error': f'Cannot load program module from: {program_path}'
            }
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if required function exists
        if not hasattr(program, 'run_multi_agent_task'):
            return {
                'runs_successfully': 0.0, 
                'score': 0.0,
                'avg_failures_per_task': 14.0,
                'error': 'Missing run_multi_agent_task function'
            }
        
        # Try to run on a simple test task
        dataset = ProgramDevDataset()
        # Set fixed seed for reproducible task sampling
        random.seed(SEED)
        sample_tasks = dataset.sample_tasks(1)
        
        if not sample_tasks:
            return {
                'runs_successfully': 0.0, 
                'score': 0.0,
                'avg_failures_per_task': 14.0,
                'error': 'No tasks in dataset'
            }
        
        task = sample_tasks[0]
        
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        # Try to run with very short timeout
        async def quick_test():
            try:
                await asyncio.wait_for(
                    program.run_multi_agent_task(
                        idea=task['description'][:100],  # Use truncated description
                        n_rounds=1,
                        log_file=log_file
                    ),
                    timeout=10
                )
                return True
            except Exception:
                return False
        
        success = asyncio.run(quick_test())
        
        # Clean up
        try:
            os.unlink(log_file)
        except:
            pass
        
        if success:
            # Calculate normalized score for 7 failures (midpoint)
            raw_score = 1.0 / (1.0 + 7.0)
            min_raw_score = 1.0 / 15.0
            max_raw_score = 1.0
            normalized_score = ((raw_score - min_raw_score) / (max_raw_score - min_raw_score)) * 100.0
            return {
                'runs_successfully': 1.0, 
                'score': float(normalized_score),
                'avg_failures_per_task': 7.0
            }
        else:
            # Calculate normalized score for 12 failures
            raw_score = 1.0 / (1.0 + 12.0)
            min_raw_score = 1.0 / 15.0
            max_raw_score = 1.0
            normalized_score = ((raw_score - min_raw_score) / (max_raw_score - min_raw_score)) * 100.0
            return {
                'runs_successfully': 0.5, 
                'score': float(normalized_score),
                'avg_failures_per_task': 12.0
            }
            
    except Exception as e:
        return {
            'runs_successfully': 0.0, 
            'score': 0.0,
            'avg_failures_per_task': 14.0,
            'error': str(e)
        }


def evaluate_stage2(program_path: str) -> Dict:
    return evaluate(program_path)


if __name__ == "__main__":
    import sys
    program_file = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    
    print(f"Evaluating {program_file}...")
    print("Stage 1:", evaluate_stage1(program_file))
    print("Stage 2:", evaluate_stage2(program_file))