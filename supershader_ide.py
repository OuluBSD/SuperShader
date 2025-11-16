#!/usr/bin/env python3
"""
SuperShader IDE - Main Application
Two-process architecture: PyQt6 IDE process manages C++ rendering process
"""

import sys
import os
import json
import subprocess
import socket
import threading
from typing import Optional, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QAction, QStatusBar, QFileDialog, QMessageBox,
    QTabWidget, QDockWidget, QLabel
)
from PyQt6.QtCore import Qt, QProcess, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QKeySequence


class CommunicationManager(QObject):
    """Manages communication between IDE and rendering processes"""
    
    # Signals for communication
    connection_established = pyqtSignal()
    connection_lost = pyqtSignal()
    message_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.socket: Optional[socket.socket] = None
        self.server_socket: Optional[socket.socket] = None
        self.client_connected = False
        self.port = 9999  # Default port for communication
        
    def start_server(self):
        """Start the communication server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('localhost', self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # Non-blocking with timeout
            
            # Start listening in a separate thread
            self.listen_thread = threading.Thread(target=self._listen_for_client, daemon=True)
            self.listen_thread.start()
            return True
        except Exception as e:
            print(f"Failed to start communication server: {e}")
            return False
    
    def _listen_for_client(self):
        """Listen for client connections in a separate thread"""
        while True:
            try:
                if self.server_socket:
                    client_socket, addr = self.server_socket.accept()
                    self.socket = client_socket
                    self.client_connected = True
                    self.connection_established.emit()
                    
                    # Start receiving messages
                    self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
                    self.receive_thread.start()
                    break
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error accepting client connection: {e}")
                break
    
    def _receive_messages(self):
        """Receive messages from the rendering process"""
        while self.client_connected and self.socket:
            try:
                data = self.socket.recv(4096)
                if not data:
                    break
                
                # Parse the received JSON message
                message_str = data.decode('utf-8').strip()
                if message_str:
                    message = json.loads(message_str)
                    self.message_received.emit(message)
            except json.JSONDecodeError:
                print(f"Received invalid JSON: {data.decode('utf-8', errors='ignore')}")
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
        
        self.client_connected = False
        self.connection_lost.emit()
    
    def send_message(self, message: Dict[str, Any]):
        """Send a message to the rendering process"""
        if self.socket and self.client_connected:
            try:
                message_str = json.dumps(message)
                self.socket.sendall(message_str.encode('utf-8'))
            except Exception as e:
                print(f"Error sending message: {e}")
                self.client_connected = False
                self.connection_lost.emit()
    
    def stop_server(self):
        """Stop the communication server"""
        self.client_connected = False
        if self.socket:
            self.socket.close()
        if self.server_socket:
            self.server_socket.close()


class RenderingProcessManager:
    """Manages the C++ rendering process"""
    
    def __init__(self, comm_manager: CommunicationManager):
        self.comm_manager = comm_manager
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        
    def start_rendering_process(self, project_path: str = ""):
        """Start the rendering process executable"""
        try:
            # For now, we'll create a placeholder script until the C++ process is built
            # In a real implementation, this would launch the C++ executable
            script_content = f'''#!/usr/bin/env python3
import json
import socket
import time
import sys

def connect_to_ide():
    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 9999))
            return sock
        except Exception as e:
            print(f"Attempt {{attempt+1}}: Could not connect to IDE: {{e}}")
            time.sleep(1)
    return None

def main():
    print("Starting rendering process...")
    sock = connect_to_ide()
    if not sock:
        print("Failed to connect to IDE")
        return

    # Send connection confirmation
    confirmation = {{"type": "connection", "status": "connected"}}
    sock.sendall(json.dumps(confirmation).encode('utf-8'))
    
    # Simulate rendering loop
    frame_count = 0
    while True:
        # Simulate frame update
        frame_info = {{"type": "frame_update", "frame": frame_count, "timestamp": time.time()}}
        try:
            sock.sendall(json.dumps(frame_info).encode('utf-8'))
        except:
            break
        frame_count += 1
        time.sleep(0.1)  # Simulate 10 FPS

if __name__ == "__main__":
    main()
'''
            # Write the placeholder script
            placeholder_path = Path("rendering_process.py")
            with open(placeholder_path, 'w') as f:
                f.write(script_content)
            
            # Start the subprocess
            self.process = subprocess.Popen([
                sys.executable, str(placeholder_path), project_path
            ])
            self.is_running = True
            print(f"Started rendering process with PID: {self.process.pid}")
            return True
        except Exception as e:
            print(f"Failed to start rendering process: {e}")
            return False
    
    def stop_rendering_process(self):
        """Stop the rendering process"""
        if self.process and self.is_running:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None
                self.is_running = False
                print("Rendering process stopped")


class SceneConfigurationPanel(QWidget):
    """Panel for configuring scene elements (entities, lighting, camera, etc.)"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Scene configuration title
        title_label = QLabel("Scene Configuration")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Entities section
        entities_label = QLabel("Entities:")
        layout.addWidget(entities_label)
        
        # Lighting section
        lighting_label = QLabel("Lighting:")
        layout.addWidget(lighting_label)
        
        # Camera section
        camera_label = QLabel("Camera:")
        layout.addWidget(camera_label)
        
        # Post-processing section
        postproc_label = QLabel("Post-Processing:")
        layout.addWidget(postproc_label)
        
        # Add stretch to push content to top
        layout.addStretch()
        
        self.setLayout(layout)


class ShaderEditorWidget(QWidget):
    """Widget for editing shader code"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Shader Editor")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Placeholder for code editor
        placeholder_label = QLabel("Shader code editor will be implemented here")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder_label)
        
        self.setLayout(layout)


class PreviewWidget(QWidget):
    """Widget for displaying rendered preview"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Preview")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Placeholder for preview
        placeholder_label = QLabel("Rendering preview will be displayed here")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("border: 1px solid gray; padding: 50px;")
        layout.addWidget(placeholder_label)
        
        self.setLayout(layout)


class SuperShaderIDE(QMainWindow):
    """Main IDE window"""
    
    def __init__(self):
        super().__init__()
        self.comm_manager = CommunicationManager()
        self.rendering_manager = RenderingProcessManager(self.comm_manager)
        self.current_project_path = ""
        
        self.init_ui()
        self.setup_connections()
        
        # Start communication server
        if self.comm_manager.start_server():
            print("Communication server started on port 9999")
        else:
            print("Failed to start communication server")
    
    def init_ui(self):
        self.setWindowTitle("SuperShader IDE")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Scene configuration
        self.scene_config_panel = SceneConfigurationPanel()
        scene_dock = QDockWidget("Scene Configuration", self)
        scene_dock.setWidget(self.scene_config_panel)
        scene_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | 
                              QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, scene_dock)
        
        # Center panel - Main content area with tabs
        center_widget = QTabWidget()
        
        # Preview tab
        self.preview_widget = PreviewWidget()
        center_widget.addTab(self.preview_widget, "Preview")
        
        # Shader editor tab
        self.shader_editor = ShaderEditorWidget()
        center_widget.addTab(self.shader_editor, "Shader Editor")
        
        main_splitter.addWidget(center_widget)
        
        # Right panel - Properties/other tools
        properties_widget = QWidget()
        properties_layout = QVBoxLayout(properties_widget)
        properties_label = QLabel("Properties\n(Will be implemented)")
        properties_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        properties_layout.addWidget(properties_label)
        properties_layout.addStretch()
        
        properties_dock = QDockWidget("Properties", self)
        properties_dock.setWidget(properties_widget)
        properties_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | 
                                   QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, properties_dock)
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_project_action = QAction('New Project', self)
        new_project_action.setShortcut(QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction('Open Project', self)
        open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        save_project_action = QAction('Save Project', self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Project menu
        project_menu = menubar.addMenu('Project')
        
        build_action = QAction('Build Project', self)
        build_action.setShortcut('F7')
        build_action.triggered.connect(self.build_project)
        project_menu.addAction(build_action)
        
        run_action = QAction('Run Project', self)
        run_action.setShortcut('F5')
        run_action.triggered.connect(self.run_project)
        project_menu.addAction(run_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_layout_action = QAction('Reset Layout', self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
    
    def create_toolbar(self):
        toolbar = self.addToolBar('Main')
        
        # Add toolbar actions
        new_action = QAction('New', self)
        new_action.triggered.connect(self.new_project)
        toolbar.addAction(new_action)
        
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_project)
        toolbar.addAction(open_action)
        
        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_project)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        build_action = QAction('Build', self)
        build_action.triggered.connect(self.build_project)
        toolbar.addAction(build_action)
        
        run_action = QAction('Run', self)
        run_action.triggered.connect(self.run_project)
        toolbar.addAction(run_action)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.comm_manager.connection_established.connect(self.on_connection_established)
        self.comm_manager.connection_lost.connect(self.on_connection_lost)
        self.comm_manager.message_received.connect(self.on_message_received)
    
    def new_project(self):
        """Create a new project"""
        project_dir = QFileDialog.getExistingDirectory(self, "New Project Location")
        if project_dir:
            project_path = Path(project_dir) / "project.json"
            project_data = {
                "name": "New Project",
                "version": "1.0",
                "settings": {
                    "rendering": {
                        "width": 1024,
                        "height": 768,
                        "target_language": "glsl"
                    }
                }
            }
            with open(project_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.current_project_path = str(project_path.parent)
            self.statusBar().showMessage(f"Created new project: {project_dir}")
            print(f"Created new project at: {project_path}")
    
    def open_project(self):
        """Open an existing project"""
        project_file, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Project Files (*.json)"
        )
        if project_file:
            self.current_project_path = str(Path(project_file).parent)
            self.statusBar().showMessage(f"Opened project: {project_file}")
            print(f"Opened project: {project_file}")
    
    def save_project(self):
        """Save the current project"""
        if self.current_project_path:
            # In a real implementation, this would save project configuration
            # For now, just show a message
            self.statusBar().showMessage(f"Project saved to: {self.current_project_path}")
        else:
            self.statusBar().showMessage("No project loaded to save")
    
    def build_project(self):
        """Build the current project"""
        if self.current_project_path:
            self.statusBar().showMessage("Building project...")
            # In a real implementation, this would initiate the build process
            print(f"Building project at: {self.current_project_path}")
            
            # Send build command to rendering process
            build_msg = {"type": "build", "project_path": self.current_project_path}
            self.comm_manager.send_message(build_msg)
            
            self.statusBar().showMessage("Build completed")
        else:
            self.statusBar().showMessage("No project loaded to build")
            QMessageBox.warning(self, "No Project", "Please open a project first.")
    
    def run_project(self):
        """Run the current project"""
        if self.current_project_path:
            self.statusBar().showMessage("Starting rendering process...")
            
            # Stop any existing rendering process
            if self.rendering_manager.is_running:
                self.rendering_manager.stop_rendering_process()
            
            # Start the rendering process
            success = self.rendering_manager.start_rendering_process(self.current_project_path)
            
            if success:
                self.statusBar().showMessage("Rendering process started")
                print("Rendering process started successfully")
            else:
                self.statusBar().showMessage("Failed to start rendering process")
                QMessageBox.critical(self, "Error", "Failed to start rendering process")
        else:
            self.statusBar().showMessage("No project loaded to run")
            QMessageBox.warning(self, "No Project", "Please open a project first.")
    
    def reset_layout(self):
        """Reset the dock widget layout to default"""
        self.tabifyDockWidget(
            self.findChild(QDockWidget, "Scene Configuration"),
            self.findChild(QDockWidget, "Properties")
        )
    
    def on_connection_established(self):
        """Handle connection to rendering process"""
        self.statusBar().showMessage("Connected to rendering process", 3000)
        print("Connected to rendering process")
    
    def on_connection_lost(self):
        """Handle disconnection from rendering process"""
        self.statusBar().showMessage("Disconnected from rendering process", 3000)
        print("Disconnected from rendering process")
        self.rendering_manager.is_running = False
    
    def on_message_received(self, message: Dict[str, Any]):
        """Handle message from rendering process"""
        msg_type = message.get("type", "unknown")
        print(f"Received message from rendering process: {msg_type}")
        
        if msg_type == "frame_update":
            # Update preview with frame info
            frame_num = message.get("frame", 0)
            self.statusBar().showMessage(f"Frame: {frame_num}", 1000)
        elif msg_type == "connection":
            status = message.get("status", "unknown")
            if status == "connected":
                self.statusBar().showMessage("Rendering process connected", 3000)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop rendering process
        if self.rendering_manager.is_running:
            self.rendering_manager.stop_rendering_process()
        
        # Stop communication server
        self.comm_manager.stop_server()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    ide = SuperShaderIDE()
    ide.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()