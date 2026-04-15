cmake_minimum_required(VERSION 3.12)
project(AutoVenvProject)

# 1. Find system Python (used to initialize virtual environment)
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter)

# 2. Set virtual environment path in project root
set(VENV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv")

# 3. Cross-platform Python path for venv
if(WIN32)
    set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
else()
    set(VENV_PYTHON "${VENV_DIR}/bin/python")
endif()

# Convert to CMake path format (important for Windows)
file(TO_CMAKE_PATH "${VENV_PYTHON}" VENV_PYTHON_PATH)

# 4. Auto-initialize venv if it doesn't exist
if(NOT EXISTS "${VENV_DIR}")
    message(STATUS ">>> First run: creating virtual environment and installing dependencies...")

    # Ensure requirements.txt exists
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")
        message(FATAL_ERROR "requirements.txt not found in project root.")
    endif()

    # 4.1 Create virtual environment
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_DIR}
        RESULT_VARIABLE VENV_RESULT
    )
    if(NOT VENV_RESULT EQUAL 0)
        message(FATAL_ERROR ">>> Virtual environment creation failed!")
    endif()

    # 4.2 Upgrade pip
    execute_process(
        COMMAND ${VENV_PYTHON_PATH} -m pip install --upgrade pip
        RESULT_VARIABLE PIP_RESULT
    )
    if(NOT PIP_RESULT EQUAL 0)
        message(FATAL_ERROR ">>> pip upgrade failed!")
    endif()

    # 4.3 Install dependencies from requirements.txt
    execute_process(
        COMMAND ${VENV_PYTHON_PATH} -m pip install -r "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
        RESULT_VARIABLE REQ_RESULT
    )
    if(NOT REQ_RESULT EQUAL 0)
        message(FATAL_ERROR ">>> Installing dependencies failed!")
    endif()

    message(STATUS ">>> Virtual environment initialized successfully!")
endif()