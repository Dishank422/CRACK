META = """Please do a meta review of the code. Check whether the description of the pull request is accurate and complete.\
Check what problem the code is trying to solve, and whether the code solves that problem. 
Check whether the code has any unintended side effects or potential issues that are not mentioned in the description.
Check whether this is the right technical direction for the project."""


OPTIMALITY = """Please do a code review of the code performance. Consider time and space complexity.
First explain the overall time and space complexity of the code, and then suggest specific improvements to make the code more optimal.
If the code is already optimal, say so and don't suggest any changes."""


MODULARITY = """Please review the code for modularity and separation of concerns. The following are some pointers for the review:
- Are there any functions that are doing too much and could be broken down into smaller functions?
- Are there any parts of the code that are tightly coupled and could be decoupled?
- Can any code be reused or abstracted to reduce duplication or the number of classes or methods?"""


EXCEPTION_HANDLING = """Please review the code for exception handling and robustness.
Please make sure that each exception provides enough context to determine the source and location of an error. 
Error messages should be informative and should mention the operation that failed and the type of failure."""


TESTING = """Please review the code tests. Make sure that the tests cover all important cases and don't miss any edge cases."""


STYLE = """Please review the code for style and readability. Check for the following:
- Are variable, function and classes named meaningfully?
- Unused variables, functions or imports?
- Are the comments sufficient and helpful? The comments should explain the why behind the code, not the what.
- Is the code style consistent with the rest of the codebase? 
Check for consistent use of whitespace, line length, and other formatting issues."""


CODE_CHECK_PROMPTS = {
    "meta": META,
    "optimality": OPTIMALITY,
    "modularity": MODULARITY,
    "exception_handling": EXCEPTION_HANDLING,
    "testing": TESTING,
    "style": STYLE,
}