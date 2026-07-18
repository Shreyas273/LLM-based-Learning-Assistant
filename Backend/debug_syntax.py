import sys
try:
    import api.routes
    print("Success")
except SyntaxError as e:
    with open("syntax_error.log", "w") as f:
        f.write(f"Line: {e.lineno}\nMsg: {e.msg}\nFilename: {e.filename}\nOffset: {e.offset}\nText: {e.text}")
except Exception as e:
    with open("syntax_error.log", "w") as f:
        f.write(f"Error: {e}")
