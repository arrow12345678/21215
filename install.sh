#!/usr/bin/env bash
#
# install.sh
# Script to install the Video Editor GUI application, its dependencies, and a desktop launcher.
#
set -euo pipefail

# --- Configuration ---
# Directory where the application will be installed
# Using .local/share is a standard place for user-specific application data.
INSTALL_DIR="${HOME}/.local/share/VideoEditorApp"
VENV_DIR="${INSTALL_DIR}/venv"

# Name of the main application script and runner script
APP_PY_FILE="video_editor_gui.py"
# RUNNER_SCRIPT is no longer needed.
ICON_FILE_PNG="video_editor_icon.png"
ICON_FILE_SVG="video_editor_icon.svg"

# Desktop file configuration
DESKTOP_FILE_NAME="VideoEditorGUI.desktop"
DESKTOP_INSTALL_DIR="${HOME}/.local/share/applications"
REQUIREMENTS_FILE="requirements.txt"

# --- Helper Functions ---
echo_info() {
    echo -e "\\033[1;34m[INFO]\\033[0m $1"
}

echo_success() {
    echo -e "\\033[1;32m[SUCCESS]\\033[0m $1"
}

echo_error() {
    echo -e "\\033[1;31m[ERROR]\\033[0m $1"
    exit 1
}

# --- Dependency Management ---
check_and_install_packages() {
    # Define required packages
    PACKAGES=("python3-venv" "python3-tk" "ffmpeg")
    MISSING_PACKAGES=()

    echo_info "Checking for required system packages..."
    if ! command -v dpkg-query >/dev/null 2>&1; then
        echo_error "dpkg not found. This script is intended for Debian-based systems (like Ubuntu)."
    fi

    for pkg in "${PACKAGES[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        echo_info "The following required packages are missing: ${MISSING_PACKAGES[*]}"
        echo_info "Sudo privileges are required to install them."
        sudo apt-get update && sudo apt-get install -y "${MISSING_PACKAGES[@]}" || { echo_error "Failed to install system dependencies. Please install them manually and re-run."; exit 1; }
    else
        echo_info "All required system packages are already installed."
    fi
}

# --- Default Icon ---
# Simple SVG icon as a fallback
create_default_icon() {
    echo_info "Creating a default SVG icon."
    cat > "${INSTALL_DIR}/${ICON_FILE_SVG}" << EOF
<svg width="64" height="64" version="1.1" viewBox="0 0 16.933 16.933" xmlns="http://www.w3.org/2000/svg">
 <rect width="16.933" height="16.933" rx="2.7932" ry="2.7932" fill="#4a4a4a" style="paint-order:stroke markers fill"/>
 <path d="m5.2917 4.2333-1.0583 8.4667h1.0583l1.0583-8.4667zm3.175 0-1.0583 8.4667h1.0583l1.0583-8.4667zm3.175 0-1.0583 8.4667h1.0583l1.0583-8.4667z" fill="#fff"/>
</svg>
EOF
}

# --- Main Installation Logic ---
main() {
    echo_info "Starting Video Editor GUI installation..."

    # 0. Check and install system dependencies
    check_and_install_packages

    # 1. Check for required script files
    if [ ! -f "$APP_PY_FILE" ]; then
        echo_error "Required file ('$APP_PY_FILE') not found."
    fi

    # 2. Create installation directory
    echo_info "Creating installation directory at: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"

    # 3. Copy application files
    echo_info "Copying application file..."
    cp "$APP_PY_FILE" "$INSTALL_DIR/"

    # 4. Handle the icon
    local icon_path
    if [ -f "$ICON_FILE_PNG" ]; then
        echo_info "User-provided PNG icon found. Copying..."
        cp "$ICON_FILE_PNG" "$INSTALL_DIR/"
        icon_path="${INSTALL_DIR}/${ICON_FILE_PNG}"
    else
        echo_info "No '$ICON_FILE_PNG' found. Creating a default SVG icon."
        create_default_icon
        icon_path="${INSTALL_DIR}/${ICON_FILE_SVG}"
    fi

    # 5. Create virtual environment and install dependencies
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo_info "Creating Python virtual environment in '${VENV_DIR}'..."
        python3 -m venv "$VENV_DIR"
        
        echo_info "Activating virtual environment and installing packages from '$REQUIREMENTS_FILE'..."
        (
            source "${VENV_DIR}/bin/activate"
            pip install --upgrade pip
            pip install -r "${REQUIREMENTS_FILE}"
        )
        if [ $? -ne 0 ]; then
            echo_error "Failed to install Python dependencies."
        fi
    else
        echo_info "Skipping Python dependencies: '$REQUIREMENTS_FILE' not found."
    fi

    # 6. Create and install the .desktop file
    echo_info "Creating and installing the desktop launcher..."
    mkdir -p "$DESKTOP_INSTALL_DIR"
    
    local exec_path="bash -c 'source ${VENV_DIR}/bin/activate && python3 ${INSTALL_DIR}/${APP_PY_FILE}'"

    DESKTOP_FILE_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=Video Editor GUI
Comment=A simple GUI for video editing tasks
Exec=${exec_path}
Icon=${icon_path}
Terminal=false
Categories=Utility;Multimedia;
StartupNotify=true
"
    echo "$DESKTOP_FILE_CONTENT" > "${DESKTOP_INSTALL_DIR}/${DESKTOP_FILE_NAME}"

    # 7. Update the desktop database
    echo_info "Updating the desktop application database..."
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$DESKTOP_INSTALL_DIR"
    else
        echo_info "Skipping database update: 'update-desktop-database' command not found."
    fi

    echo_success "Installation completed successfully!"
    echo_info "You can now find 'Video Editor GUI' in your application menu."
    echo_info "To uninstall, run: rm -rf ${INSTALL_DIR} ${DESKTOP_INSTALL_DIR}/${DESKTOP_FILE_NAME}"
}

# Run the main function
main 
