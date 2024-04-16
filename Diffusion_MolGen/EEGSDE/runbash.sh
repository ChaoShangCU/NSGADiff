#!/bin/bash

log_file="my_log_file.txt"  # Specify the name of the log file

# Source the configuration file
source config.sh

for i in {21..30}; do
   #Task 1: Change the value in utils.py
  newcode="property_norms[property_key]['mean']=-$i.00"
  if grep -q "$code_to_modify_utils" /home/chao/EEGSDE/qm9/utils.py; then
    sed -i "s/$code_to_modify_utils/$newcode/" /home/chao/EEGSDE/qm9/utils.py

    # Print the result on the screen
    echo "Task 1: Modified utils.py for iteration $i"
  else
    echo "Original code not found in utils.py for iteration $i"
  fi
  # Task 2: Generate code for rdkit_functions.py with the updated value of i
  new_filename="valid_molecules_Cv_${i}.txt"
  new_code="with open(f'$new_filename', 'w') as file:"

  # Modify rdkit_functions.py with the dynamically generated code
  if grep -q "$code_to_modify_rdkit" /home/chao/EEGSDE/qm9/rdkit_functions.py; then
    sed -i "s/$code_to_modify_rdkit/$new_code/" /home/chao/EEGSDE/qm9/rdkit_functions.py

    # Print the result on the screen
    echo "Task 2: Modified rdkit_functions.py for iteration $i"
  else
    echo "Original code not found in rdkit_functions.py for iteration $i"
  fi

  # Update the code_to_modify_rdkit variable manually
  code_to_modify_rdkit="$new_code"

  # Task 3: Run the command
  # $python_script
  # Print the result on the screen
  echo "Task 3: Ran the command for iteration $i"
done

