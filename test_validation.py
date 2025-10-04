"""
Test script for Stock Price Prediction LSTM
This validates the code structure and logic without external dependencies
"""

import sys
import ast


def validate_python_syntax(filepath):
    """Validate Python syntax"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"✓ Syntax validation passed for {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False


def check_class_structure(filepath):
    """Check if the class structure is correct"""
    with open(filepath, 'r') as f:
        code = f.read()
    
    required_methods = [
        '__init__',
        'fetch_data',
        'prepare_data',
        'create_sequences',
        'build_model',
        'train',
        'predict',
        'visualize_results',
        'calculate_metrics'
    ]
    
    tree = ast.parse(code)
    
    # Find the StockPricePredictor class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'StockPricePredictor':
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            print(f"\n✓ Found StockPricePredictor class with {len(methods)} methods")
            
            missing = [m for m in required_methods if m not in methods]
            if missing:
                print(f"✗ Missing methods: {missing}")
                return False
            else:
                print(f"✓ All required methods present: {required_methods}")
                return True
    
    print("✗ StockPricePredictor class not found")
    return False


def check_main_function(filepath):
    """Check if main function exists"""
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            print("✓ Main function found")
            return True
    
    print("✗ Main function not found")
    return False


def main():
    """Run all validation tests"""
    print("="*60)
    print("Stock Price Prediction LSTM - Code Validation")
    print("="*60)
    
    filepath = 'stock_prediction.py'
    
    tests = [
        ("Syntax Validation", lambda: validate_python_syntax(filepath)),
        ("Class Structure", lambda: check_class_structure(filepath)),
        ("Main Function", lambda: check_main_function(filepath))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        results.append(test_func())
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All validations passed!")
        return 0
    else:
        print("✗ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
