#!/usr/bin/env python3
"""
Test runner for RAG Document Search System
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, get_logger
import logging

# Set up logging
loggers = setup_logging(log_level=logging.INFO, log_file="test_runner.log")
logger = get_logger('test_runner')

def run_test_file(test_file):
    """Run a single test file."""
    try:
        logger.info(f"🧪 Running test: {test_file}")
        
        # Run the test file
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(test_file))
        
        if result.returncode == 0:
            logger.info(f"✅ {test_file} passed")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ {test_file} failed")
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {test_file}: {e}")
        return False

def run_all_tests():
    """Run all test files in the tests directory."""
    logger.info("="*60)
    logger.info("🧪 RUNNING ALL TESTS")
    logger.info("="*60)
    
    # Get all test files
    tests_dir = Path(__file__).parent
    test_files = [
        "test_logging.py",
        "test_search.py", 
        "test_search_fixed.py",
        "test_system.py"
    ]
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            if run_test_file(str(test_path)):
                passed += 1
            else:
                failed += 1
        else:
            logger.warning(f"⚠️ Test file not found: {test_file}")
    
    # Summary
    logger.info("="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📝 Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed!")
        return False

def run_specific_test(test_name):
    """Run a specific test by name."""
    tests_dir = Path(__file__).parent
    test_file = tests_dir / f"{test_name}.py"
    
    if test_file.exists():
        logger.info(f"🧪 Running specific test: {test_name}")
        return run_test_file(str(test_file))
    else:
        logger.error(f"❌ Test file not found: {test_name}.py")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 