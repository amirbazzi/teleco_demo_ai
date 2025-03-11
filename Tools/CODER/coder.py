from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from langchain_core.tools import tool
import plotly
import json
# Navigate up two levels to the project root and add it to the sys.path
import plotly.io as pio
# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."], return_direct=True
):
    
    """Use this to execute python code and do math.  
    If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    try:
        result = repl.run(code)
        figure_path = 'figure.json'
        print("DEBUG GENERATE REPL CODE ============ ")
        print(code)
        print("DEBUG GENERATE REPL FIGURE ============ ")
        print(result)





        pio.write_json(result, figure_path)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result