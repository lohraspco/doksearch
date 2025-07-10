#!/usr/bin/env python3
"""
Script to restart the Streamlit app with updated code
"""

import subprocess
import sys
import os
import time
import signal
from logging_config import setup_logging, get_logger
import logging

# Set up logging
loggers = setup_logging(log_level=logging.INFO, log_file="restart_app.log")
logger = get_logger('restart_app')

def find_streamlit_process():
    """Find running Streamlit process."""
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq streamlit.exe'], 
                              capture_output=True, text=True, shell=True)
        if 'streamlit.exe' in result.stdout:
            return True
    except:
        pass
    
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'streamlit' in result.stdout:
            return True
    except:
        pass
    
    return False

def kill_streamlit_process():
    """Kill running Streamlit process."""
    try:
        # Windows
        subprocess.run(['taskkill', '/F', '/IM', 'streamlit.exe'], 
                      capture_output=True, shell=True)
        logger.info("Killed Streamlit process (Windows)")
        return True
    except:
        pass
    
    try:
        # Unix/Linux/Mac
        subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)
        logger.info("Killed Streamlit process (Unix)")
        return True
    except:
        pass
    
    return False

def start_streamlit_app():
    """Start the Streamlit app."""
    try:
        logger.info("Starting Streamlit app...")
        
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'advanced_chat.py',
            '--server.port', '8501',
            '--server.headless', 'true'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Streamlit started with PID: {process.pid}")
        
        # Wait a moment for the app to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ Streamlit app started successfully")
            logger.info("🌐 App should be available at: http://localhost:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Streamlit failed to start")
            logger.error(f"STDOUT: {stdout.decode()}")
            logger.error(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"Error starting Streamlit: {e}")
        return None

def main():
    """Main function to restart the app."""
    logger.info("🔄 Restarting Streamlit RAG Chat App")
    
    # Check if Streamlit is running
    if find_streamlit_process():
        logger.info("Found running Streamlit process, killing it...")
        kill_streamlit_process()
        time.sleep(2)  # Wait for process to fully terminate
    
    # Start the app
    process = start_streamlit_app()
    
    if process:
        logger.info("✅ App restart completed successfully!")
        logger.info("📝 Check logs/restart_app.log for details")
        logger.info("📝 Check logs/advanced_chat.log for app logs")
        
        # Keep the script running to maintain the process
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
            process.terminate()
            process.wait()
    else:
        logger.error("❌ Failed to restart the app")
        sys.exit(1)

if __name__ == "__main__":
    main() 