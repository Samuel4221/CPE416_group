#!/bin/bash

# Script to copy files to polkit and udev directories - only copy if not exists
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Directories
POLKIT_DIR="/etc/polkit-1/localauthority/50-local.d"
UDEV_DIR="/etc/udev/rules.d"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

log_success() {
    echo -e "${CYAN}[SUCCESS]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root. Use sudo."
        exit 1
    fi
}

# Set up the 'pwm' group and add the current $USER
setup_pwm_group() {
    print_separator
    log_info "Setting up PWM group access..."
    
    # Create pwm group
    if ! getent group pwm >/dev/null; then
        groupadd pwm
        log_success "Created 'pwm' group"
    else
        log_info "'pwm' group already exists"
    fi
    
    # Add user to pwm group
    current_user=$(logname 2>/dev/null || echo $SUDO_USER || echo $USER)
    if ! groups $current_user | grep -q '\bpwm\b'; then
        usermod -a -G pwm $current_user
        log_success "Added $current_user to pwm group"
    else
        log_info "$current_user is already in pwm group"
    fi
}

# Check if source files exist
check_source_files() {
    local source_dir="$1"
    local file_type="$2"
    
    if [[ ! -d "$source_dir" ]]; then
        log_error "Source directory $source_dir does not exist!"
        return 1
    fi
    
    if [[ -z "$(ls -A "$source_dir" 2>/dev/null)" ]]; then
        log_error "No files found in $source_dir!"
        return 1
    fi
    
    local file_count=$(ls -1 "$source_dir" | wc -l)
    log_info "Found $file_count $file_type files to process"
    return 0
}

# Copy files only if they don't exist
copy_files_if_missing() {
    local source_dir="$1"
    local dest_dir="$2"
    local file_type="$3"
    
    log_info "Processing ${CYAN}$file_type${NC} files from ${BLUE}$source_dir${NC} to ${BLUE}$dest_dir${NC}"
    
    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"
    
    local copied=0
    local skipped=0
    
    for source_file in "$source_dir"/*; do
        if [[ -f "$source_file" ]]; then
            local filename=$(basename "$source_file")
            local dest_file="$dest_dir/$filename"
            
            if [[ -f "$dest_file" ]]; then
                echo -e "${GREEN}✓${NC} ${filename} ${YELLOW}already exists${NC} - skipping"
                ((skipped++))
            else
                echo -e "${GREEN}+${NC} ${filename} ${CYAN}does not exist${NC} - copying"
                cp -v "$source_file" "$dest_dir/"
                chmod 644 "$dest_dir/$filename"
                chown root:root "$dest_dir/$filename"
                ((copied++))
            fi
        fi
    done
    
    echo -e "${GREEN}●${NC} ${file_type} summary: ${CYAN}${copied} new${NC}, ${YELLOW}${skipped} already existed${NC}"
    
    if [[ $copied -gt 0 ]]; then
        return 0  # Files were copied
    else
        return 2  # No files copied (all already exist)
    fi
}

# Reload services only if files were copied
reload_services() {
    log_info "Reloading system services..."
    
    echo -e "${BLUE}↻${NC} Reloading udev rules..."
    if udevadm control --reload-rules && udevadm trigger; then
        echo -e "${GREEN}✓${NC} Udev rules reloaded successfully"
    else
        log_warning "Failed to reload udev rules"
    fi
    
    # Reload polkit (if systemd is available)
    if command -v systemctl >/dev/null 2>&1; then
        echo -e "${BLUE}↻${NC} Reloading polkit service..."
        if systemctl try-reload-or-restart polkit.service >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Polkit service reloaded"
        else
            log_warning "Polkit service reload may not be needed"
        fi
    fi
}

# Print a separator line
print_separator() {
    echo -e "${BLUE}──────────────────────────────────────────────────────────────${NC}"
}

# Main function
main() {
    local POLKIT_SOURCE="./polkit_files"  # Change this to your source directory
    local UDEV_SOURCE="./udev_files"      # Change this to your source directory
    
    print_separator
    log_success "Starting configuration file copy script"
    print_separator
    
    check_root
    setup_pwm_group
    print_separator
    
    local changes_made=false
    
    # Process polkit files
    if check_source_files "$POLKIT_SOURCE" "polkit"; then
        print_separator
        if copy_files_if_missing "$POLKIT_SOURCE" "$POLKIT_DIR" "polkit"; then
            changes_made=true
        fi
        print_separator
    else
        log_warning "Skipping polkit file processing"
        print_separator
    fi
    
    # Process udev files
    if check_source_files "$UDEV_SOURCE" "udev"; then
        if copy_files_if_missing "$UDEV_SOURCE" "$UDEV_DIR" "udev"; then
            changes_made=true
        fi
        print_separator
    else
        log_warning "Skipping udev file processing"
        print_separator
    fi
    
    # Only reload services if new files were copied
    if [[ "$changes_made" = true ]]; then
        log_info "New files were copied - reloading services"
        reload_services
    else
        log_info "No new files copied - services not reloaded"
    fi
    
    print_separator
    log_success "Script completed successfully!"
    log_success "You should reboot the system and then try running the 'simple_pwm.cpp' code to make sure everything is OK!"
    print_separator
}

# Usage instructions
usage() {
    echo -e "${GREEN}Usage:${NC} sudo $0"
    echo ""
    echo -e "${CYAN}This script copies configuration files to:${NC}"
    echo -e "  - ${BLUE}$POLKIT_DIR${NC}"
    echo -e "  - ${BLUE}$UDEV_DIR${NC}"
    echo ""
    echo -e "${YELLOW}File handling:${NC}"
    echo -e "  - ${GREEN}Only copies files that don't already exist${NC}"
    echo -e "  - ${YELLOW}Skips files that already exist${NC}"
    echo -e "  - ${RED}Never overwrites existing files${NC}"
    echo ""
    echo -e "${CYAN}Before running:${NC}"
    echo -e "1. Place polkit files in ${BLUE}./polkit_files/${NC}"
    echo -e "2. Place udev files in ${BLUE}./udev_files/${NC}"
    echo -e "3. Make this script executable: ${GREEN}chmod +x $0${NC}"
    echo -e "4. Run with: ${GREEN}sudo $0${NC}"
}

# Check if help requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"