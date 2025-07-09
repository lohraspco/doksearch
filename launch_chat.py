#!/usr/bin/env python3
"""
Chat Interface Launcher
=======================

This script helps you choose and launch the appropriate chat interface
for your RAG document system.
"""

import subprocess
import sys
import os

def main():
    print("🤖 RAG Document System - Chat Interface Launcher")
    print("=" * 50)
    print()
    print("Choose your chat interface:")
    print()
    print("1. 💬 Simple Chat UI - Clean and straightforward")
    print("2. 🤖 Advanced Chat UI - Modern with animations and features")
    print("3. 📊 Full Dashboard - Complete system with all features")
    print("4. 🧪 Test System - Run tests first")
    print("5. 🔧 Setup System - Install dependencies")
    print("6. ❌ Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            print("🚀 Launching Simple Chat UI...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "chat_ui.py"])
            break
        elif choice == "2":
            print("🚀 Launching Advanced Chat UI...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "advanced_chat.py"])
            break
        elif choice == "3":
            print("🚀 Launching Full Dashboard...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
            break
        elif choice == "4":
            print("🧪 Running system tests...")
            subprocess.run([sys.executable, "test_system.py"])
            break
        elif choice == "5":
            print("🔧 Setting up system...")
            subprocess.run([sys.executable, "setup.py"])
            break
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main() 