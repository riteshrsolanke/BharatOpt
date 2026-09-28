#pragma once
#include "bharatopt/lp/iteration_record.hpp"
#include "bharatopt/milp/node.hpp"
#include <fstream>
#include <vector>

namespace bharatopt {

class JSONExporter {
public:
    static void export_data(const std::string& filepath, 
                            const std::vector<LPIterationRecord>& lp_iters,
                            const std::vector<MILPNode>& milp_nodes) {
        std::ofstream out(filepath);
        if (!out.is_open()) return;

        out << "{\n";
        
        // Export LP Iterations
        out << "  \"lp_iterations\": [\n";
        for (size_t i = 0; i < lp_iters.size(); ++i) {
            const auto& rec = lp_iters[i];
            out << "    {\n";
            out << "      \"iteration\": " << rec.iteration << ",\n";
            out << "      \"entering_variable\": " << rec.entering_variable << ",\n";
            out << "      \"leaving_variable\": " << rec.leaving_variable << ",\n";
            out << "      \"pivot_element\": " << rec.pivot_element << ",\n";
            out << "      \"objective_value\": " << rec.objective_value << ",\n";
            out << "      \"current_solution\": [";
            for (size_t j = 0; j < rec.current_solution.size(); ++j) {
                out << rec.current_solution[j] << (j + 1 < rec.current_solution.size() ? ", " : "");
            }
            out << "]\n";
            out << "    }" << (i + 1 < lp_iters.size() ? "," : "") << "\n";
        }
        out << "  ],\n";

        // Export MILP Nodes
        out << "  \"milp_nodes\": [\n";
        for (size_t i = 0; i < milp_nodes.size(); ++i) {
            const auto& n = milp_nodes[i];
            out << "    {\n";
            out << "      \"id\": " << n.id << ",\n";
            out << "      \"depth\": " << n.depth << ",\n";
            out << "      \"parent_id\": " << n.parent_id << ",\n";
            out << "      \"bound\": " << n.bound << ",\n";
            out << "      \"incumbent\": " << n.incumbent_at_creation << ",\n";
            out << "      \"branching_variable\": " << n.branching_variable << ",\n";
            out << "      \"branching_value\": " << n.branching_value << ",\n";
            out << "      \"branch_direction\": \"" << n.branch_direction << "\",\n";
            
            std::string status_str = "OPEN";
            if (n.status == NodeStatus::INFEASIBLE) status_str = "INFEASIBLE";
            if (n.status == NodeStatus::INTEGER) status_str = "INTEGER";
            if (n.status == NodeStatus::PRUNED_BY_BOUND) status_str = "PRUNED_BY_BOUND";
            if (n.status == NodeStatus::COMPLETED) status_str = "COMPLETED";
            
            out << "      \"status\": \"" << status_str << "\"\n";
            out << "    }" << (i + 1 < milp_nodes.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }
};

} // namespace bharatopt
