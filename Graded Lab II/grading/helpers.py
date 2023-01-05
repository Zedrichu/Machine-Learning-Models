""" Helpers used for generating data, testing and grading.
"""

# 3rd party.
import numpy as np
from IPython.display import display, Markdown, Latex

# Python std.
import os

# Project files.
from grading.parameters import grading_params


# Paths to grading data and solutions
path_grade_data = os.path.join('grading', 'data')
path_grade_solutions = os.path.join('solutions', 'data')


class IsolatedRNG:
    """ Use fixed random seed locally. Saves and restores Random Number
    Generator state. Use as context guard with 'with'.

    Args:
        seed (int): RNG seed.

    Example:

    with isolatedRNG():
        do your random thing
    """
    def __init__(self, seed=28008):
        self.seed = seed
        self.rng_state = None

    def __enter__(self):
        self.rng_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.rng_state)


def get_data(exercise_id):
    """ Loads data required by the exercise passed as `exercise_id`. Exercise must be
    registered in `grading_params` dictionary.

    Args:
        exercise_id (str): Name of the exercise.

    Returns:
        grade_data (dict): Loaded grading data.
    """
    grade_data = None

    # Check existence of the exercise_id.
    if not exercise_id in grading_params:
        raise Exception(f'"{exercise_id}" not found within "grading_params" items '
                        f'{list(grading_params.keys())}')

    # Load data.
    data_path = exercise_id + '.npz'
    with np.load(os.path.join(path_grade_data, data_path), allow_pickle=True) as data_file:
        grade_data = dict(data_file.items())

    return grade_data


# TODO - add path to Moodle!!!
def show_submission_instructions(scope):
    """ Shows submission instructions.

    Args:
        scope (dict): Local variables.
    """
    submission_path = get_submission_path(scope)
    moodle_path = '>>> TODO <<<'
    markdown = '#### You have reached the end of the tests.\n\n'
    markdown += '### [**Click here to go to moodle to upload your ' \
                'submission**]({}).\n\n'.format(moodle_path)
    markdown += 'Please submit the following files ONLY:\n'
    markdown += '- `graded_exercise_1.ipynb`\n'
    markdown += '- `{}` (automatically generated)\n'.format(submission_path)
    display(Markdown(markdown))


def resolve(var_name, scope):
    """ Finds the given symbol (a function or a variable) in the scope. Reports
    error if not found.

    Args:
        var_name (str): Name of the symbol to be found.
        scope (dict): Local variables.

    Returns:
        Found variable or a function.
    """
    variable = scope.get(var_name)
    fail_msg = "'{}' is not defined in the current scope".format(var_name)
    assert variable is not None, fail_msg
    return variable


def get_submission_path(scope):
    """ Returns the filename for the submission for the current student.
    Also validates the student's sciper number.

    Args:
        scope (dict): Local variables.

    Returns:
        str: Name of the answer file.
    """
    sciper_number = resolve('sciper_number', scope)

    # ensure sciper_number has right format.
    sciper_number = str(sciper_number)
    fail_msg = 'Please correct your sciper number, otherwise you will not be ' \
               'able to generate your results file.'
    assert str.isdigit(sciper_number) and \
           len(sciper_number) == len('123456'), fail_msg

    answers_file_path = 'answers_{}.npz'.format(sciper_number)

    return answers_file_path


def register_answer(name, answer, scope):
    """ Adds the `answer` to answers dictionary for grading.

    NOTE: Names in the dictionary have to be unique.

    Args:
        name (str): Name of the exercise to save results for.
        answer: The result.
        scope (dict): Local variables.
    """
    answers_file_path = get_submission_path(scope)

    if 'answers' not in scope:
        scope['answers'] = dict()
        if os.path.isfile(answers_file_path):
            scope['answers'].update(np.load(answers_file_path, allow_pickle=True).items())
    scope['answers'][name] = answer

    np.savez(answers_file_path, **scope['answers'])


def compare_np_arrays(computed, expected, *, compare_types=True,
                      compare_shapes=True, compare_values=True, varname=None):
    """ Common function to compare two numpy arrays.
        - Validates that computed object is a numpy array.
        - Compares computed and expected types.
        - Compares computed and expected sizes.
        - Compares computed and expected values.

    Args:
        computed (np.array): Computed data.
        expected (np.array): Expected data.
        compare_types (bool): Whether to compare array data types.
        compare_shapes (bool): Whether to compare array shapes.
        compare_values (bool): Whether to compare array values.
        varname (str or None): Name to print out.
    """
    # Ensure computed is a numpy array.
    fail_msg = 'Computed {} is not a numpy array. It is of type {}'. \
        format(varname or 'object', type(computed))
    assert isinstance(computed, np.ndarray), fail_msg

    # Ensure arrays are of the same type.
    if compare_types:
        fail_msg = 'Computed {} is of type {}, but {} was expected'. \
            format(varname or 'array', computed.dtype, expected.dtype)
        assert computed.dtype == expected.dtype, fail_msg

    # Ensure arrays are of the same shape.
    if compare_shapes:
        fail_msg = 'Computed {} dimensions are {}, but {} was expected'. \
            format(varname or 'array', computed.shape, expected.shape)
        assert computed.shape == expected.shape, fail_msg

    # Ensure arrays have similar values.
    if compare_values:
        fail_msg = 'Computed {} does not have the expected values. Make sure ' \
                   'your computation is correct.'.format(varname or 'array')
        assert np.all(np.isclose(computed, expected, atol=1e-3)), fail_msg


def compare_int(computed, expected, *, compare_types=True,
                      compare_values=True, varname=None):
    """ Common function to compare two numpy arrays.
        - Validates that computed object is a numpy array.
        - Compares computed and expected types.
        - Compares computed and expected sizes.
        - Compares computed and expected values.

    Args:
        computed (np.array): Computed data.
        expected (np.array): Expected data.
        compare_types (bool): Whether to compare array data types.
        compare_shapes (bool): Whether to compare array shapes.
        compare_values (bool): Whether to compare array values.
        varname (str or None): Name to print out.
    """
    # Ensure computed is a numpy array.
    fail_msg = 'Computed {} is not a int. It is of type {}'. \
        format(varname or 'object', type(computed))
    assert isinstance(computed, np.int), fail_msg

    # Ensure arrays are of the same type.
    # if compare_types:
    #     fail_msg = 'Computed {} is of type {}, but {} was expected'. \
    #         format(varname or 'int', computed.dtype, expected.dtype)
    #     assert computed.dtype == expected.dtype, fail_msg


    # Ensure arrays have similar values.
    if compare_values:
        fail_msg = 'Computed {} does not have the expected values. Make sure ' \
                   'your computation is correct.'.format(varname or 'int')
        assert np.all(np.isclose(computed, expected, atol=1e-3)), fail_msg


