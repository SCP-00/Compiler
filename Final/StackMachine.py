# StackMachine.py (Interprets IR code)

import sys
import struct 
from typing import List, Tuple, Any, Dict, Optional, Callable
from Error import ErrorHandler
# Assuming IRProgramRepresentation and IRFunctionRepresentation are defined
# (e.g., in IRCodeGenerator.py or a shared ir_defs.py)
# For standalone testing, we might need to redefine them or import.
# Let's assume they can be imported from where IRCodeGenerator is.
try:
    from IRCodeGenerator import IRProgramRepresentation, IRFunctionRepresentation 
except ImportError:
    # Fallback dummy definitions if running StackMachine standalone for testing
    # without full pipeline and IRCodeGenerator.py is not in path.
    print("Warning: Could not import IR representation classes from IRCodeGenerator. Using dummies.", file=sys.stderr)
    class IRFunctionRepresentation:
        def __init__(self, name, params, return_ir_type, is_imported=False, code=None, locals_info=None):
            self.name = name; self.params = params or []; self.return_ir_type = return_ir_type
            self.is_imported = is_imported; self.code = code or []; self.locals = locals_info or {}
    class IRProgramRepresentation:
        def __init__(self): self.globals = {}; self.functions = {}


IRInstruction = Tuple[str, ...]
IR_Type = str 
StackValue = Any 
StackEntry = Tuple[IR_Type, StackValue]


class StackMachine:
    def __init__(self, error_handler: ErrorHandler, memory_size_bytes: int = 1024 * 1024):
        self.error_handler: ErrorHandler = error_handler
        self.stack: List[StackEntry] = []
        self.memory: bytearray = bytearray(memory_size_bytes)
        
        self.globals: Dict[str, StackEntry] = {} 
        
        self.call_stack_return_pcs: List[int] = []
        self.locals_env_stack: List[Dict[str, StackEntry]] = [{}] # Global/base execution environment

        # functions will store metadata needed for CALL, extracted from IRFunctionRepresentation
        self.functions_meta: Dict[str, Dict[str, Any]] = {} 
        self.program_ir: List[IRInstruction] = [] # Flattened IR code from all non-imported functions
        self.pc: int = 0 
        self.running: bool = False

    def _push_locals_scope(self): self.locals_env_stack.append({})
    def _pop_locals_scope(self):
        if len(self.locals_env_stack) > 1: self.locals_env_stack.pop()

    def _current_locals_scope(self) -> Dict[str, StackEntry]: return self.locals_env_stack[-1]

    def _assign_local(self, name: str, typed_value: StackEntry, current_pc_for_error: int):
        # Assign to current function's frame (innermost scope)
        self._current_locals_scope()[name] = typed_value

    def _lookup_local(self, name: str, current_pc_for_error: int) -> Optional[StackEntry]:
        # Search from current scope outwards (only relevant if supporting nested scopes for locals,
        # but GoX probably doesn't have closures that would require this deep search for locals)
        # For simple functions, only the current scope (top of locals_env_stack) matters.
        env = self._current_locals_scope()
        if name in env:
            return env[name]
        # If not in current, check if it's a parameter (which are also in current_locals_scope for this model)
        # This lookup logic might need refinement based on how parameters vs locals are handled by IR Generator.
        # For now, if not found in current scope, it's an error.
        self.error_handler.add_interpretation_error(f"Runtime: Local variable or parameter '{name}' not found.", current_pc_for_error)
        return None

    def _get_default_ir_value(self, ir_type: IR_Type) -> StackValue:
        if ir_type == 'I': return 0
        if ir_type == 'F': return 0.0
        # Add for 'S' (string address/handle) or other IR types if they exist
        return None # Should not happen for 'V' (void)

    def load_ir(self, ir_program_repr: IRProgramRepresentation):
        """Loads IR from the IRProgramRepresentation object."""
        self.program_ir = []
        self.functions_meta = {}
        self.globals = {}
        current_pc_offset = 0

        for name, global_info in ir_program_repr.globals.items():
            ir_type = global_info["type"]
            self.globals[name] = (ir_type, self._get_default_ir_value(ir_type))

        for func_name, func_repr in ir_program_repr.functions.items():
            self.functions_meta[func_name] = {
                'name': func_repr.name,
                'start_pc': current_pc_offset if not func_repr.is_imported else -1, # -1 for imported
                'params': func_repr.params, # List of (name, ir_type)
                'locals_info': func_repr.locals, # Dict of name -> ir_type
                'return_ir_type': func_repr.return_ir_type,
                'is_imported': func_repr.is_imported,
            }
            if not func_repr.is_imported:
                self.program_ir.extend(func_repr.code)
                current_pc_offset += len(func_repr.code)
        # print(f"VM Loaded {len(self.program_ir)} instructions. Functions: {list(self.functions_meta.keys())}")


    def run(self, entry_function_name: str = "_init"):
        if not self.program_ir and not self.functions_meta.get(entry_function_name, {}).get('is_imported'):
            entry_func_meta = self.functions_meta.get(entry_function_name)
            if not entry_func_meta or (not entry_func_meta.get('is_imported') and not self.program_ir) :
                 self.error_handler.add_interpretation_error("No program loaded or empty entry function.", None); return

        entry_func_meta = self.functions_meta.get(entry_function_name)
        if not entry_func_meta:
            self.error_handler.add_interpretation_error(f"Entry function '{entry_function_name}' not found.", None); return
        
        # If _init is imported, it's a special case not handled here (e.g. linking external C main)
        if entry_func_meta['is_imported']:
             self.error_handler.add_interpretation_error(f"Cannot directly run imported function '{entry_function_name}' as program entry.", None); return


        self.pc = entry_func_meta['start_pc']
        self.running = True
        # Initial scope for _init already at self.locals_env_stack[0]
        # If _init has parameters (it shouldn't), they would need to be pushed here.

        while self.running and 0 <= self.pc < len(self.program_ir):
            instr = self.program_ir[self.pc]
            opcode = instr[0]
            args = list(instr[1:])
            
            # print(f"VMExec: PC={self.pc:03d} | StackDepth={len(self.stack)} | Op: {opcode} {args} | StackTop: {self.stack[-1] if self.stack else '[]'}")
            
            current_instr_pc = self.pc 
            self.pc += 1 

            method = getattr(self, f"op_{opcode}", self.op_UNKNOWN)
            try: method(*args)
            except IndexError: 
                self.error_handler.add_interpretation_error(f"Stack underflow during '{opcode}'.", current_instr_pc)
                self.running = False
            except TypeError as te: 
                self.error_handler.add_interpretation_error(f"VM Type error or arity mismatch for '{opcode}': {te}", current_instr_pc)
                self.running = False
            except ZeroDivisionError as zde:
                self.error_handler.add_interpretation_error(f"Runtime error: {str(zde)} during '{opcode}'.", current_instr_pc)
                self.running = False
            except Exception as e: 
                self.error_handler.add_interpretation_error(f"Unexpected runtime error during '{opcode}': {type(e).__name__} - {str(e)}", current_instr_pc)
                self.running = False
        
        if len(self.locals_env_stack) > 1 : self._pop_locals_scope()

    def op_UNKNOWN(self, *args):
        bad_opcode = "UnknownOpcode"
        if self.pc > 0 and self.pc <= len(self.program_ir):
            bad_opcode = self.program_ir[self.pc-1][0]
        self.error_handler.add_interpretation_error(f"Unknown IR opcode: {bad_opcode}", self.pc -1)
        self.running = False

    # --- Stack Operations ---
    def _pop_stack(self, op_name_for_error: str) -> StackEntry:
        if not self.stack: raise IndexError(f"{op_name_for_error}: Stack underflow.")
        return self.stack.pop()

    def _pop_typed_val(self, expected_ir_type: IR_Type, op_name: str) -> StackValue:
        ir_type, val = self._pop_stack(op_name)
        if ir_type != expected_ir_type:
            raise TypeError(f"{op_name}: Expected IR type '{expected_ir_type}', got '{ir_type}'.")
        return val

    def op_CONSTI(self, value: int): self.stack.append(('I', int(value)))
    def op_CONSTF(self, value: float): self.stack.append(('F', float(value)))

    def op_ADDI(self): self.stack.append(('I', self._pop_typed_val('I', "ADDI_rhs") + self._pop_typed_val('I', "ADDI_lhs")))
    def op_SUBI(self): r=self._pop_typed_val('I',"SUBI_rhs");l=self._pop_typed_val('I',"SUBI_lhs"); self.stack.append(('I', l - r))
    def op_MULI(self): self.stack.append(('I', self._pop_typed_val('I', "MULI_rhs") * self._pop_typed_val('I', "MULI_lhs")))
    def op_DIVI(self): 
        r=self._pop_typed_val('I',"DIVI_rhs");l=self._pop_typed_val('I',"DIVI_lhs")
        if r == 0: raise ZeroDivisionError("Integer division by zero")
        self.stack.append(('I', l // r))
    
    # def op_MODI(self): # Example, if you add MODI to your IR spec
    #     r=self._pop_typed_val('I',"MODI_rhs");l=self._pop_typed_val('I',"MODI_lhs")
    #     if r == 0: raise ZeroDivisionError("Integer modulo by zero")
    #     self.stack.append(('I', l % r))

    def op_ADDF(self): self.stack.append(('F', self._pop_typed_val('F', "ADDF_rhs") + self._pop_typed_val('F', "ADDF_lhs")))
    def op_SUBF(self): r=self._pop_typed_val('F',"SUBF_rhs");l=self._pop_typed_val('F',"SUBF_lhs"); self.stack.append(('F', l - r))
    def op_MULF(self): self.stack.append(('F', self._pop_typed_val('F', "MULF_rhs") * self._pop_typed_val('F', "MULF_lhs")))
    def op_DIVF(self):
        r=self._pop_typed_val('F',"DIVF_rhs");l=self._pop_typed_val('F',"DIVF_lhs")
        if r == 0.0: raise ZeroDivisionError("Float division by zero")
        self.stack.append(('F', l / r))

    def _compare_op(self, py_operator: Callable[[Any, Any], bool], ir_operand_type: IR_Type, op_name: str):
        r = self._pop_typed_val(ir_operand_type, f"{op_name}_rhs")
        l = self._pop_typed_val(ir_operand_type, f"{op_name}_lhs")
        self.stack.append(('I', 1 if py_operator(l, r) else 0))

    def op_LTI(self): self._compare_op(lambda a,b: a < b,  'I', "LTI")
    def op_LEI(self): self._compare_op(lambda a,b: a <= b, 'I', "LEI")
    def op_GTI(self): self._compare_op(lambda a,b: a > b,  'I', "GTI")
    def op_GEI(self): self._compare_op(lambda a,b: a >= b, 'I', "GEI")
    def op_EQI(self): self._compare_op(lambda a,b: a == b, 'I', "EQI")
    def op_NEI(self): self._compare_op(lambda a,b: a != b, 'I', "NEI")
    
    def op_LTF(self): self._compare_op(lambda a,b: a < b,  'F', "LTF")
    def op_LEF(self): self._compare_op(lambda a,b: a <= b, 'F', "LEF")
    def op_GTF(self): self._compare_op(lambda a,b: a > b,  'F', "GTF")
    def op_GEF(self): self._compare_op(lambda a,b: a >= b, 'F', "GEF")
    def op_EQF(self): self._compare_op(lambda a,b: a == b, 'F', "EQF")
    def op_NEF(self): self._compare_op(lambda a,b: a != b, 'F', "NEF")

    def op_ANDI(self): 
        r=self._pop_typed_val('I',"ANDI_rhs");l=self._pop_typed_val('I',"ANDI_lhs")
        self.stack.append(('I', 1 if (l != 0 and r != 0) else 0))
    def op_ORI(self): 
        r=self._pop_typed_val('I',"ORI_rhs");l=self._pop_typed_val('I',"ORI_lhs")
        self.stack.append(('I', 1 if (l != 0 or r != 0) else 0))
    def op_NOTI(self): # Assumes input is 0 or 1
        val = self._pop_typed_val('I', "NOTI_operand")
        self.stack.append(('I', 1 if val == 0 else 0))

    def op_PRINTI(self): print(self._pop_typed_val('I', "PRINTI"), end='')
    def op_PRINTF(self): print(self._pop_typed_val('F', "PRINTF"), end='')
    def op_PRINTB(self): 
        char_code = self._pop_typed_val('I', "PRINTB")
        try: print(chr(char_code), end='')
        except ValueError: self.error_handler.add_interpretation_error(f"PRINTB: Value {char_code} invalid for chr().", self.pc-1)

    def op_ITOF(self): self.stack.append(('F', float(self._pop_typed_val('I', "ITOF"))))
    def op_FTOI(self): self.stack.append(('I', int(self._pop_typed_val('F', "FTOI"))))

    def op_LOCAL_GET(self, name: str):
        val_tuple = self._lookup_local(name, self.pc-1)
        if val_tuple: self.stack.append(val_tuple)
        else: self.running = False
    def op_LOCAL_SET(self, name: str):
        val_tuple = self._pop_stack(f"LOCAL_SET '{name}'")
        self._assign_local(name, val_tuple, self.pc-1)
        
    def op_GLOBAL_GET(self, name: str):
        if name in self.globals: self.stack.append(self.globals[name])
        else: 
            self.error_handler.add_interpretation_error(f"Runtime: Global variable '{name}' not found.", self.pc-1)
            self.running = False
    def op_GLOBAL_SET(self, name: str):
        val_tuple = self._pop_stack(f"GLOBAL_SET '{name}'")
        # Type check against global declaration if available (from IRModule.globals[name].type)
        # For simplicity, this basic VM assumes type correctness is mostly by IR generator.
        self.globals[name] = val_tuple

    def op_CALL(self, func_name: str):
        target_func_meta = self.functions_meta.get(func_name)
        if not target_func_meta:
            self.error_handler.add_interpretation_error(f"Runtime: CALL to undefined IR function '{func_name}'.", self.pc-1); self.running = False; return

        if target_func_meta.get('is_imported', False):
            if func_name == "put_image": # Example imported function
                num_expected_params = 3 # base, width, height (all 'I')
                if len(self.stack) < num_expected_params: raise IndexError(f"{func_name} call: stack underflow for arguments.")
                # Args are on stack: ..., arg1, arg2, arg3 (top). Pop in reverse order of push.
                h_val = self._pop_typed_val('I', "put_image.height")
                w_val = self._pop_typed_val('I', "put_image.width")
                b_val = self._pop_typed_val('I', "put_image.base")
                # print(f"[VM STUB CALL] put_image(base={b_val}, width={w_val}, height={h_val})")
                self.stack.append(('I', 0)) # put_image returns int
                return # PC already advanced for next instruction after CALL
            else:
                 self.error_handler.add_interpretation_error(f"Runtime: CALL to unhandled imported function '{func_name}'.", self.pc-1); self.running = False; return

        self.call_stack_return_pcs.append(self.pc) 
        self._push_locals_scope() 
        
        num_params = len(target_func_meta['params'])
        if len(self.stack) < num_params:
            raise IndexError(f"CALL {func_name}: Not enough arguments on stack for parameters.")
        
        # Parameters are typically pushed by caller in order (param1, param2, ... paramN)
        # So stack top is paramN. We pop them in reverse order of declaration to match.
        args_from_stack_for_params = self.stack[-num_params:]
        self.stack = self.stack[:-num_params] 

        for (param_name, param_ir_type), arg_stack_entry in zip(target_func_meta['params'], args_from_stack_for_params):
            # Check type compatibility between arg_stack_entry[0] (arg IR type) and param_ir_type
            if arg_stack_entry[0] != param_ir_type:
                 # Basic check: allow 'I' (from bool/char) to 'I' (for int param)
                if not (param_ir_type == 'I' and arg_stack_entry[0] == 'I'): # Stricter check if needed
                    raise TypeError(f"CALL {func_name}: Type mismatch for param '{param_name}'. Expected IR type '{param_ir_type}', got '{arg_stack_entry[0]}'.")
            self._assign_local(param_name, arg_stack_entry, self.pc-1)

        # Initialize declared local variables to default values for this function
        for local_name, local_ir_type in target_func_meta['locals_info'].items():
            default_val = self._get_default_ir_value(local_ir_type)
            self._assign_local(local_name, (local_ir_type, default_val), self.pc-1)

        self.pc = target_func_meta['start_pc']

    def op_RET(self):
        # Return value (if any) is expected to be on top of the stack.
        # It is left there for the caller to use.
        self._pop_locals_scope() 
        if not self.call_stack_return_pcs:
            self.running = False # Return from _init (or main if it's the entry)
            return
        self.pc = self.call_stack_return_pcs.pop()

    def _ensure_memory_access(self, address: int, num_bytes: int, op_name: str):
        if not (0 <= address and address + num_bytes <= len(self.memory)):
            self.error_handler.add_interpretation_error(
                f"{op_name}: Memory access out of bounds (addr={address}, len={num_bytes}, capacity={len(self.memory)}). Did you GROW memory?", self.pc-1
            )
            raise IndexError("Memory access out of bounds")

    def op_PEEKB(self): 
        addr = self._pop_typed_val('I', "PEEKB_addr")
        self._ensure_memory_access(addr, 1, "PEEKB")
        self.stack.append(('I', int(self.memory[addr])))
    def op_POKEB(self): 
        addr = self._pop_typed_val('I', "POKEB_addr") # Address is pushed after value by IR Generator
        val = self._pop_typed_val('I', "POKEB_val")   # Value is deeper in stack
        if not (0 <= val <= 255): raise ValueError(f"POKEB value {val} not in byte range [0-255].")
        self._ensure_memory_access(addr, 1, "POKEB")
        self.memory[addr] = val

    def op_PEEKI(self):
        addr = self._pop_typed_val('I', "PEEKI_addr")
        self._ensure_memory_access(addr, 4, "PEEKI") 
        val = int.from_bytes(self.memory[addr:addr+4], byteorder='little', signed=True)
        self.stack.append(('I', val))
    def op_POKEI(self):
        addr = self._pop_typed_val('I', "POKEI_addr")
        val = self._pop_typed_val('I', "POKEI_val")
        self._ensure_memory_access(addr, 4, "POKEI")
        self.memory[addr:addr+4] = val.to_bytes(4, byteorder='little', signed=True)

    def op_PEEKF(self):
        addr = self._pop_typed_val('I', "PEEKF_addr")
        self._ensure_memory_access(addr, 8, "PEEKF") # Assuming 8-byte float (double)
        val, = struct.unpack('<d', self.memory[addr:addr+8]) # '<d' is little-endian double
        self.stack.append(('F', val))
    def op_POKEF(self):
        addr = self._pop_typed_val('I', "POKEF_addr")
        val = self._pop_typed_val('F', "POKEF_val")
        self._ensure_memory_access(addr, 8, "POKEF")
        self.memory[addr:addr+8] = struct.pack('<d', val)

    def op_GROW(self): 
        num_units_to_grow = self._pop_typed_val('I', "GROW_size")
        if num_units_to_grow < 0:
            self.error_handler.add_interpretation_error("GROW size cannot be negative.", self.pc-1)
            self.stack.append(('I', len(self.memory))); return # Return current capacity on error
        
        # Your IR spec "retorna nuevo tamaño". This usually means new *total* capacity.
        # For bytearray, we might need to extend it.
        current_len = len(self.memory)
        # If num_units_to_grow is the *additional* size:
        required_len = current_len + num_units_to_grow
        if required_len > current_len:
            try:
                self.memory.extend(bytearray(num_units_to_grow))
            except MemoryError:
                self.error_handler.add_interpretation_error(f"Out of memory trying to GROW by {num_units_to_grow} bytes.", self.pc-1)
                self.running = False
                self.stack.append(('I', current_len)); return # Push old capacity on error

        self.stack.append(('I', len(self.memory))) # Push new total capacity in bytes

    # --- Structured Control Flow (IF, LOOP, etc. from ircode.py) ---
    # These require careful handling of PC and potentially a control stack.
    _control_flow_markers_stack = [] # To find matching ENDIF/ENDLOOP for structured ops

    def op_IF(self): 
        cond_val = self._pop_typed_val('I', "IF_cond") # bool as 0 or 1
        if cond_val == 0: # If false, jump to matching ELSE or ENDIF
            nesting_level = 0
            jump_pc = self.pc # Start search from instruction after IF
            while jump_pc < len(self.program_ir):
                opc = self.program_ir[jump_pc][0]
                if opc == 'IF': nesting_level += 1
                elif opc == 'ENDIF':
                    if nesting_level == 0: self.pc = jump_pc + 1; return
                    nesting_level -= 1
                elif opc == 'ELSE':
                    if nesting_level == 0: self.pc = jump_pc + 1; return
                jump_pc += 1
            self.error_handler.add_interpretation_error("IF without matching ENDIF/ELSE.", self.pc-1); self.running = False
        # If true, PC just continues to the next instruction (consequence block)

    def op_ELSE(self): # Unconditional jump to matching ENDIF
        nesting_level = 0
        jump_pc = self.pc 
        while jump_pc < len(self.program_ir):
            opc = self.program_ir[jump_pc][0]
            # Note: IFs inside this ELSE block would increment nesting_level
            if opc == 'IF': nesting_level +=1
            elif opc == 'ENDIF':
                if nesting_level == 0: self.pc = jump_pc + 1; return
                nesting_level -=1
            jump_pc +=1
        self.error_handler.add_interpretation_error("ELSE without matching ENDIF.", self.pc-1); self.running = False

    def op_ENDIF(self): pass # PC has already advanced past it or jumped past it.

    def op_LOOP(self):
        # Mark the start of this loop for CONTINUE and for ENDLOOP to jump back.
        self._control_flow_markers_stack.append({'type': 'LOOP', 'start_pc': self.pc -1 }) # PC of LOOP instr

    def op_ENDLOOP(self):
        if not self._control_flow_markers_stack or self._control_flow_markers_stack[-1]['type'] != 'LOOP':
            self.error_handler.add_interpretation_error("ENDLOOP without matching LOOP.", self.pc-1); self.running = False; return
        
        loop_info = self._control_flow_markers_stack.pop() # Pop this loop's marker
        self.pc = loop_info['start_pc'] # Jump back to the LOOP instruction itself

    def op_CBREAK(self): 
        cond_val = self._pop_typed_val('I', "CBREAK_cond") # bool as 0 or 1
        if cond_val != 0: # If true, break from current loop
            if not self._control_flow_markers_stack or self._control_flow_markers_stack[-1]['type'] != 'LOOP':
                self.error_handler.add_interpretation_error("CBREAK not meaningfully inside a LOOP structure for VM.", self.pc-1); self.running = False; return
            
            # Find the ENDLOOP matching the current LOOP on control_flow_markers_stack
            current_loop_start_pc = self._control_flow_markers_stack[-1]['start_pc']
            nesting_level = 0
            jump_pc = current_loop_start_pc # Start search from the LOOP instruction
            
            found_matching_endloop = False
            while jump_pc < len(self.program_ir):
                opc = self.program_ir[jump_pc][0]
                if opc == 'LOOP' and jump_pc > current_loop_start_pc : # Inner loop starts
                    # Only count loops that started *at or after* our current loop start
                    if self.program_ir[jump_pc] == self.program_ir[current_loop_start_pc]: # If it's the same loop marker
                        pass # This is our loop beginning
                    elif jump_pc > current_loop_start_pc : # An inner loop
                         nesting_level += 1
                elif opc == 'ENDLOOP':
                    if nesting_level == 0: # This ENDLOOP matches our current_loop_start_pc
                        self.pc = jump_pc + 1 # Jump to instruction AFTER ENDLOOP
                        self._control_flow_markers_stack.pop() # This loop is now exited
                        found_matching_endloop = True
                        break
                    else: # This is an ENDLOOP for an inner loop
                        nesting_level -= 1
                jump_pc += 1
            
            if not found_matching_endloop:
                self.error_handler.add_interpretation_error("CBREAK could not find matching ENDLOOP.", self.pc-1); self.running = False
        # If condition is false, CBREAK does nothing, execution continues

    def op_CONTINUE(self):
        if not self._control_flow_markers_stack or self._control_flow_markers_stack[-1]['type'] != 'LOOP':
            self.error_handler.add_interpretation_error("CONTINUE not meaningfully inside a LOOP structure for VM.", self.pc-1); self.running = False; return
        self.pc = self._control_flow_markers_stack[-1]['start_pc'] # Jump to start of current LOOP

# --- Main Driver ---
def main():
    import os
    # These would be your actual compiler phase outputs
    from Lexer import tokenize 
    from Parser import Parser
    from SemanticAnalyzer import SemanticAnalyzer
    from IRCodeGenerator import IRCodeGenerator # This now generates the IR

    if len(sys.argv) < 2:
        print("Usage: python StackMachine.py <gox_file_to_compile_and_run>")
        sys.exit(1)

    source_file_path = sys.argv[1]
    if not (os.path.exists(source_file_path) and source_file_path.endswith('.gox')):
        print(f"Error: Source file not found or not a .gox file: '{source_file_path}'")
        sys.exit(1)
    
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f: content = f.read()
    except Exception as e: print(f"Error reading file '{source_file_path}': {e}"); sys.exit(1)

    print(f"\n--- Compiling and Running IR for: {source_file_path} ---")
    error_handler = ErrorHandler()

    print("1. Lexical Analysis...")
    tokens = tokenize(content, error_handler)
    if error_handler.has_errors(): error_handler.report_errors(); sys.exit(1)
    print("   Lexical Analysis successful.")

    print("2. Parsing...")
    parser = Parser(tokens, error_handler)
    ast_root = parser.parse()
    if error_handler.has_errors() or not ast_root:
        if not ast_root: print("   Parser failed to produce AST.")
        error_handler.report_errors(); sys.exit(1)
    print("   Parsing successful.")

    print("3. Semantic Analysis...")
    semantic_analyzer = SemanticAnalyzer(error_handler)
    semantic_analyzer.analyze(ast_root)
    if error_handler.has_errors():
        print("\nSEMANTIC ERRORS found:")
        error_handler.report_errors(); sys.exit(1)
    print("   Semantic Analysis successful.")

    print("4. IR Code Generation...")
    ir_generator = IRCodeGenerator(error_handler)
    # ir_generator.print_ir_to_console = True # Enable to see IR during generation
    ir_program_representation = ir_generator.generate_ir(ast_root)
    if error_handler.has_errors() or not ir_program_representation:
        if not ir_program_representation: print("   IR Generator failed to produce IR.")
        print("\nIR GENERATION ERRORS found:")
        error_handler.report_errors(); sys.exit(1)
    print("   IR Code Generation successful. (IR was printed during generation)")

    # Save IR to a file (optional)
    # ir_output_file = source_file_path.replace(".gox", ".ir.json")
    # import json
    # try:
    #     with open(ir_output_file, 'w') as f:
    #         json.dump(ir_program_representation.to_dict(), f, indent=2)
    #     print(f"   IR Representation saved to: {ir_output_file}")
    # except Exception as e:
    #     print(f"   Could not save IR to JSON: {e}")


    print("\n5. Stack Machine Execution...")
    print("--- Program Output (from StackMachine) ---")
    vm = StackMachine(error_handler)
    vm.load_ir(ir_program_representation) # Load the IR data structure
    
    if error_handler.has_errors(): # Errors during IR loading (e.g. bad structure)
        print("\nERRORS during IR loading into StackMachine:")
        error_handler.report_errors(); sys.exit(1)
    
    vm.run(entry_function_name="_init") # Execute from _init

    if error_handler.has_errors():
        print("\n--- STACK MACHINE EXECUTION ERRORS ---")
        error_handler.report_errors()
    else:
        print("\n--- Stack Machine execution finished successfully. ---")
        # if vm.stack: print(f"Final VM stack (should ideally be clean or hold _init result): {vm.stack}")


if __name__ == '__main__':
    main()
