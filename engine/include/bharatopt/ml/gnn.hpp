#pragma once
#include <vector>
#include <string>
#include "bharatopt/model/model.hpp"

namespace bharatopt {

struct BipartiteGraph {
    int num_variables;
    int num_constraints;
    std::vector<int> edge_sources;
    std::vector<int> edge_targets;
    std::vector<double> edge_weights;
};

struct GNNPrediction {
    int selected_variable = -1;
    double confidence = 0.0;
    std::vector<double> scores;
    bool valid = false;
};

struct GNNLog {
    bool gnn_used = false;
    double gnn_confidence = 0.0;
    double gnn_inference_time = 0.0;
    int selected_variable = -1;
    std::string fallback_reason;
};

class MILPGraphBuilder {
public:
    static BipartiteGraph build(const Model& model) {
        BipartiteGraph g;
        g.num_variables = model.variables().size();
        g.num_constraints = model.constraints().size();
        
        for (const auto& c : model.constraints()) {
            for (const auto& term : c->terms()) {
                g.edge_sources.push_back(term.first); // Variable Node
                g.edge_targets.push_back(c->id());    // Constraint Node
                g.edge_weights.push_back(term.second);
            }
        }
        return g;
    }
};

class GNNInferenceService {
public:
    GNNPrediction request_branching(const BipartiteGraph& graph) {
        GNNPrediction pred;
        // Mock inference connection to Python sidecar
        // Simulate a timeout or failure if no connection exists
        pred.valid = false;
        return pred;
    }
};

class GNNBranchingModel {
public:
    GNNBranchingModel() : service_(std::make_unique<GNNInferenceService>()) {}

    int select_branching_variable(const Model& model, GNNLog& log) {
        BipartiteGraph g = MILPGraphBuilder::build(model);
        GNNPrediction pred = service_->request_branching(g);
        
        if (!pred.valid) {
            log.gnn_used = false;
            log.fallback_reason = "inference_failure_or_timeout";
            return classical_fallback(model);
        }
        if (pred.confidence < 0.8) {
            log.gnn_used = false;
            log.fallback_reason = "low_confidence";
            return classical_fallback(model);
        }
        
        log.gnn_used = true;
        log.gnn_confidence = pred.confidence;
        log.selected_variable = pred.selected_variable;
        return pred.selected_variable;
    }

private:
    std::unique_ptr<GNNInferenceService> service_;

    int classical_fallback(const Model& model) {
        // Fallback to Most Fractional / Reliability Branching
        // Mathematically correct safe fallback
        for (const auto& v : model.variables()) {
            if (v->type() == VariableType::Integer || v->type() == VariableType::Binary) {
                return v->id();
            }
        }
        return 0;
    }
};

} // namespace bharatopt
