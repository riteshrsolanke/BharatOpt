#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include "bharatopt/model/model.hpp"

namespace bharatopt {

class MPSParser {
public:
    static Model parse(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open MPS file: " + filepath);
        }

        Model model("ParsedModel");
        std::string line;
        
        enum class Section { NONE, NAME, ROWS, COLUMNS, RHS, BOUNDS };
        Section current_section = Section::NONE;
        
        std::unordered_map<std::string, int> row_map;
        std::unordered_map<std::string, int> var_map;
        std::string obj_name = "";

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            // Section headers start at the first character
            if (line[0] != ' ' && line[0] != '\t') {
                std::istringstream iss(line);
                std::string header;
                iss >> header;
                
                if (header == "NAME") {
                    current_section = Section::NAME;
                    std::string name;
                    if (iss >> name) model = Model(name);
                } else if (header == "ROWS") {
                    current_section = Section::ROWS;
                } else if (header == "COLUMNS") {
                    current_section = Section::COLUMNS;
                } else if (header == "RHS") {
                    current_section = Section::RHS;
                } else if (header == "BOUNDS") {
                    current_section = Section::BOUNDS;
                } else if (header == "ENDATA") {
                    break;
                }
                continue;
            }

            std::istringstream iss(line);
            if (current_section == Section::ROWS) {
                std::string type, name;
                if (iss >> type >> name) {
                    if (type == "N") {
                        obj_name = name; // Objective row
                    } else {
                        ConstraintType c_type = ConstraintType::Equal;
                        if (type == "L") c_type = ConstraintType::LessEqual;
                        if (type == "G") c_type = ConstraintType::GreaterEqual;
                        int c_id = model.add_constraint(name, c_type, 0.0);
                        row_map[name] = c_id;
                    }
                }
            } else if (current_section == Section::COLUMNS) {
                std::string col_name, row_name;
                double val;
                if (iss >> col_name >> row_name >> val) {
                    if (var_map.find(col_name) == var_map.end()) {
                        var_map[col_name] = model.add_variable(col_name, VariableType::Continuous, {0.0, 1e20});
                    }
                    int v_id = var_map[col_name];
                    
                    if (row_name == obj_name) {
                        model.objective().add_term(v_id, val);
                    } else if (row_map.find(row_name) != row_map.end()) {
                        model.constraints()[row_map[row_name]]->add_term(v_id, val);
                    }
                    
                    // MPS columns can have two pairs per line, checking for second pair
                    std::string row_name2;
                    double val2;
                    if (iss >> row_name2 >> val2) {
                        if (row_name2 == obj_name) {
                            model.objective().add_term(v_id, val2);
                        } else if (row_map.find(row_name2) != row_map.end()) {
                            model.constraints()[row_map[row_name2]]->add_term(v_id, val2);
                        }
                    }
                }
            } else if (current_section == Section::RHS) {
                std::string rhs_name, row_name;
                double val;
                if (iss >> rhs_name >> row_name >> val) {
                    if (row_map.find(row_name) != row_map.end()) {
                        // Assuming constraints were constructed with 0.0 RHS initially
                        // We would ideally call model.constraints()[row_map[row_name]]->set_rhs(val);
                        // But since it's immutable in our current API, this is a placeholder.
                    }
                    // Handle second pair if exists
                    std::string row_name2;
                    double val2;
                    if (iss >> row_name2 >> val2) {
                         if (row_map.find(row_name2) != row_map.end()) {
                            // Placeholder
                        }
                    }
                }
            }
            // Skipping BOUNDS detail for basic prototype
        }
        return model;
    }
};

} // namespace bharatopt
