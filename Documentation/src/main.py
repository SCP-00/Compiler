# main.py
import tkinter as tk
from tkinter import filedialog
import subprocess
import os
import sys
from Reader_script import read_script
from Parser import Parser
from Semantic_cheking import SemanticChecker
from Interpreter import Interpreter

def run_reader_script(filename):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reader_path = os.path.join(script_dir, 'Reader_script.py')
        # Usa el mismo intérprete de Python que ejecuta la GUI
        result = subprocess.run(
            [sys.executable, reader_path, filename],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)

def open_file_dialog():
    filename = filedialog.askopenfilename(
        initialdir=".",
        title="Select a file",
        filetypes=(("Gox files", "*.gox"), ("all files", "*.*"))
    )
    if filename:
        run_reader_script(filename)

def main():
    script = read_script(sys.argv[1])
    ast = Parser(script).parse()
    SemanticChecker().visit(ast)
    result = Interpreter().visit_Program(ast)
    return 0

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Compiler GUI")

    button = tk.Button(root, text="Select Gox File", command=open_file_dialog)
    button.pack(pady=20)

    root.mainloop()
    sys.exit(main())