#!/usr/bin/env python3
"""
Setup script for RAG Document System
====================================

This script helps users set up the RAG system by:
1. Creating a new virtual environment
2. Installing dependencies
3. Creating necessary directories
4. Setting up environment variables
5. Running initial tests

Usage:
    python setup.py [venv_name]
    python setup.py my_venv
    python setup.py --help
"""

import os
import sys
import subprocess
import shutil
import platform
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup RAG Document System with virtual environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py              # Use default 'venv' name
  python setup.py my_rag_env   # Use custom environment name
  python setup.py --help       # Show this help message
        """
    )
    
    parser.add_argument(
        'venv_name',
        nargs='?',
        default='venv',
        help='Name of the virtual environment to create (default: venv)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of existing virtual environment'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running tests after installation'
    )
    
    return parser.parse_args()

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_virtual_environment(venv_name, force=False):
    """Create a new virtual environment."""
    if os.path.exists(venv_name):
        if force:
            print(f"🗑️  Removing existing virtual environment '{venv_name}'...")
            shutil.rmtree(venv_name)
        else:
            print(f"⚠️  Virtual environment '{venv_name}' already exists")
            response = input("Do you want to recreate it? (y/N): ").strip().lower()
            if response == 'y':
                print(f"🗑️  Removing existing virtual environment...")
                shutil.rmtree(venv_name)
            else:
                print(f"✅ Using existing virtual environment: {venv_name}")
                return venv_name
    
    print(f"🔄 Creating virtual environment: {venv_name}")
    
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print(f"✅ Virtual environment created: {venv_name}")
        
        # Verify the environment was created correctly
        if verify_virtual_environment(venv_name):
            return venv_name
        else:
            print(f"❌ Virtual environment verification failed for: {venv_name}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None

def verify_virtual_environment(venv_name):
    """Verify that the virtual environment was created correctly."""
    python_executable = get_python_executable(venv_name)
    pip_executable = get_pip_executable(venv_name)
    
    # Check if Python executable exists
    if not os.path.exists(python_executable):
        print(f"   ❌ Python executable not found: {python_executable}")
        return False
    
    # Check if pip executable exists
    if not os.path.exists(pip_executable):
        print(f"   ❌ Pip executable not found: {pip_executable}")
        return False
    
    # Test if Python can run
    try:
        result = subprocess.run(
            [python_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   ✅ Python version: {result.stdout.strip()}")
        else:
            print(f"   ❌ Python test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Python test failed: {e}")
        return False
    
    # Test if pip can run
    try:
        result = subprocess.run(
            [pip_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   ✅ Pip version: {result.stdout.strip()}")
        else:
            print(f"   ❌ Pip test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Pip test failed: {e}")
        return False
    
    return True

def get_python_executable(venv_name):
    """Get the Python executable path for the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_name, "Scripts", "python.exe")
    else:
        return os.path.join(venv_name, "bin", "python")

def get_pip_executable(venv_name):
    """Get the pip executable path for the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_name, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_name, "bin", "pip")

def upgrade_pip(venv_name):
    """Upgrade pip in the virtual environment."""
    print("🔄 Upgrading pip...")
    pip_executable = get_pip_executable(venv_name)
    
    try:
        subprocess.check_call([pip_executable, "install", "--upgrade", "pip"])
        print("✅ Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to upgrade pip: {e}")
        print("   This is usually not critical, continuing with existing pip version...")
        return True  # Continue anyway, as this is not critical
    except FileNotFoundError:
        print(f"⚠️  Pip executable not found at: {pip_executable}")
        print("   This might be a virtual environment issue, but continuing...")
        return True  # Continue anyway

def install_dependencies(venv_name):
    """Install required dependencies in the virtual environment."""
    print("📦 Installing dependencies...")
    pip_executable = get_pip_executable(venv_name)
    
    try:
        # First, try to install with --upgrade to get latest compatible versions
        print("   Trying with latest compatible versions...")
        subprocess.check_call([pip_executable, "install", "--upgrade", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  First attempt failed: {e}")
        print("   Trying alternative installation method...")
        
        try:
            # Try installing core packages first, then the rest
            core_packages = [
                "numpy>=1.21.0",
                "pandas>=2.0.0",
                "requests>=2.31.0",
                "python-dotenv>=1.0.0"
            ]
            
            print("   Installing core packages first...")
            for package in core_packages:
                subprocess.check_call([pip_executable, "install", package])
            
            print("   Installing remaining packages...")
            subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully (alternative method)")
            return True
            
        except subprocess.CalledProcessError as e2:
            print(f"❌ Alternative installation also failed: {e2}")
            print(f"   Pip executable: {pip_executable}")
            print("   This might be due to system-specific issues.")
            
            # Offer minimal installation
            if os.path.exists("requirements_minimal.txt"):
                print("   Trying minimal installation...")
                try:
                    subprocess.check_call([pip_executable, "install", "-r", "requirements_minimal.txt"])
                    print("✅ Minimal dependencies installed successfully")
                    print("   Note: Some advanced features may not be available")
                    return True
                except subprocess.CalledProcessError as e3:
                    print(f"❌ Minimal installation also failed: {e3}")
                    
                    # Try basic installation as last resort
                    if os.path.exists("requirements_basic.txt"):
                        print("   Trying basic installation (no version constraints)...")
                        try:
                            subprocess.check_call([pip_executable, "install", "-r", "requirements_basic.txt"])
                            print("✅ Basic dependencies installed successfully")
                            print("   Note: Using latest available versions")
                            return True
                        except subprocess.CalledProcessError as e4:
                            print(f"❌ Basic installation also failed: {e4}")
            
            print("   You can try installing manually:")
            print(f"   {pip_executable} install -r requirements.txt")
            print("   Or try the minimal version:")
            print(f"   {pip_executable} install -r requirements_minimal.txt")
            print("   Or try the basic version:")
            print(f"   {pip_executable} install -r requirements_basic.txt")
            return False
    except FileNotFoundError:
        print(f"❌ Pip executable not found at: {pip_executable}")
        print("   Please check if the virtual environment was created correctly.")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "downloads",
        "processed",
        "chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created: {directory}/")

def setup_environment():
    """Set up environment variables."""
    print("🔧 Setting up environment...")
    
    env_file = ".env"
    env_example = "env_example.txt"
    
    if not os.path.exists(env_file) and os.path.exists(env_example):
        shutil.copy(env_example, env_file)
        print(f"✅ Created {env_file} from {env_example}")
        print("   Please edit .env and add your OpenAI API key if needed")
    elif os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
    else:
        print(f"⚠️  {env_example} not found, skipping environment setup")

def test_imports(venv_name):
    """Test if all required modules can be imported in the virtual environment."""
    print("🧪 Testing imports...")
    
    python_executable = get_python_executable(venv_name)
    required_modules = [
        "langchain",
        "chromadb",
        "sentence_transformers",
        "PyPDF2",
        "docx",
        "streamlit",
        "openai",
        "transformers",
        "torch"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            result = subprocess.run(
                [python_executable, "-c", f"import {module}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"   ✅ {module}")
            else:
                print(f"   ❌ {module}")
                failed_imports.append(module)
        except Exception as e:
            print(f"   ❌ {module} - {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def run_basic_test(venv_name):
    """Run a basic test of the system in the virtual environment."""
    print("🧪 Running basic test...")
    
    python_executable = get_python_executable(venv_name)
    
    try:
        # Test RAG system initialization
        test_script = """
import sys
sys.path.append('.')
from rag_system import RAGSystem
rag_system = RAGSystem()
stats = rag_system.get_system_stats()
print("✅ RAG system initialized successfully")
print(f"   OpenAI Available: {stats['openai_available']}")
print(f"   Local LLM Available: {stats['local_llm_available']}")
print(f"   Local Embeddings: {stats['local_embeddings']}")
"""
        
        result = subprocess.run(
            [python_executable, "-c", test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Basic test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def create_activation_script(venv_name):
    """Create activation script for easy environment activation."""
    print("📝 Creating activation script...")
    
    if platform.system() == "Windows":
        script_content = f"""@echo off
echo Activating RAG Document System virtual environment...
call {venv_name}\\Scripts\\activate.bat
echo.
echo Virtual environment activated!
echo You can now run:
echo   python main.py --help
echo   streamlit run advanced_chat.py
echo   python model_manager.py
echo.
cmd /k
"""
        script_file = "activate_rag.bat"
    else:
        script_content = f"""#!/bin/bash
echo "Activating RAG Document System virtual environment..."
source {venv_name}/bin/activate
echo ""
echo "Virtual environment activated!"
echo "You can now run:"
echo "  python main.py --help"
echo "  streamlit run advanced_chat.py"
echo "  python model_manager.py"
echo ""
exec $SHELL
"""
        script_file = "activate_rag.sh"
        # Make executable
        os.chmod(script_file, 0o755)
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created activation script: {script_file}")

def create_run_scripts(venv_name):
    """Create convenient run scripts for the virtual environment."""
    print("📝 Creating run scripts...")
    
    python_executable = get_python_executable(venv_name)
    
    # Create run script for main.py
    if platform.system() == "Windows":
        run_main = f"""@echo off
call {venv_name}\\Scripts\\activate.bat
python main.py %*
"""
        run_chat = f"""@echo off
call {venv_name}\\Scripts\\activate.bat
streamlit run advanced_chat.py %*
"""
        run_manager = f"""@echo off
call {venv_name}\\Scripts\\activate.bat
python model_manager.py %*
"""
    else:
        run_main = f"""#!/bin/bash
source {venv_name}/bin/activate
python main.py "$@"
"""
        run_chat = f"""#!/bin/bash
source {venv_name}/bin/activate
streamlit run advanced_chat.py "$@"
"""
        run_manager = f"""#!/bin/bash
source {venv_name}/bin/activate
python model_manager.py "$@"
"""
    
    # Write scripts
    scripts = [
        ("run_rag.bat" if platform.system() == "Windows" else "run_rag.sh", run_main),
        ("run_chat.bat" if platform.system() == "Windows" else "run_chat.sh", run_chat),
        ("run_manager.bat" if platform.system() == "Windows" else "run_manager.sh", run_manager)
    ]
    
    for script_name, content in scripts:
        with open(script_name, 'w') as f:
            f.write(content)
        
        if platform.system() != "Windows":
            os.chmod(script_name, 0o755)
        
        print(f"   ✅ Created: {script_name}")

def main():
    """Main setup function."""
    args = parse_arguments()
    
    print("🚀 RAG Document System Setup")
    print("=" * 40)
    print(f"This will create a new virtual environment '{args.venv_name}' and install all dependencies.")
    print()
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Create virtual environment
    venv_name = create_virtual_environment(args.venv_name, args.force)
    if not venv_name:
        print("❌ Setup failed at virtual environment creation")
        return
    
    # Step 3: Upgrade pip (optional, continue even if it fails)
    upgrade_pip(venv_name)
    
    # Step 4: Install dependencies
    if not install_dependencies(venv_name):
        print("❌ Setup failed at dependency installation")
        return
    
    # Step 5: Create directories
    create_directories()
    
    # Step 6: Setup environment
    setup_environment()
    
    # Step 7: Test imports
    if not test_imports(venv_name):
        print("❌ Setup failed at import testing")
        return
    
    # Step 8: Run basic test (unless skipped)
    if not args.skip_tests:
        if not run_basic_test(venv_name):
            print("❌ Setup failed at basic testing")
            return
    else:
        print("⏭️  Skipping tests as requested")
    
    # Step 9: Create activation and run scripts
    create_activation_script(venv_name)
    create_run_scripts(venv_name)
    
    print("\n✅ Setup completed successfully!")
    print("\n🎉 Your RAG Document System is ready!")
    print(f"\n📁 Virtual environment: {venv_name}/")
    print("\n📖 Next steps:")
    print("  1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"     {venv_name}\\Scripts\\activate")
        print("     or run: activate_rag.bat")
    else:
        print(f"     source {venv_name}/bin/activate")
        print("     or run: ./activate_rag.sh")
    
    print("\n  2. Configure your models:")
    print("     python model_manager.py")
    
    print("\n  3. Test the system:")
    print("     python test_system.py")
    
    print("\n  4. Launch the chat interface:")
    print("     streamlit run advanced_chat.py")
    
    print("\n  5. Or use the convenient run scripts:")
    if platform.system() == "Windows":
        print("     run_rag.bat --help")
        print("     run_chat.bat")
        print("     run_manager.bat")
    else:
        print("     ./run_rag.sh --help")
        print("     ./run_chat.sh")
        print("     ./run_manager.sh")
    
    print("\n💡 Tip: The virtual environment keeps your RAG system isolated from other Python projects!")

if __name__ == "__main__":
    main() 