#pragma once
#include <string>
#include <vector>
#include "bharatopt/model/model.hpp"
#include "bharatopt/router/router.hpp"
#include "bharatopt/cert/certificate.hpp"

namespace bharatopt {
namespace refinery {

struct Feedstock {
    std::string name;
    double available_quantity;
    double purchase_cost;
    double sulfur_content;
};

struct Product {
    std::string name;
    double demand;
    double selling_price;
    double max_sulfur;
};

struct UnitOperation {
    std::string name;
    double capacity;
    double operating_cost;
    double energy_cost;
    std::vector<double> yields;
};

struct EconomicAnalysis {
    double revenue = 0.0;
    double raw_material_cost = 0.0;
    double operating_cost = 0.0;
    double energy_cost = 0.0;
    double estimated_margin = 0.0;
    std::vector<std::pair<std::string, double>> binding_constraints;
};

class RefineryModelBuilder {
public:
    static Model build(const std::vector<Feedstock>& feeds,
                       const std::vector<Product>& products,
                       const std::vector<UnitOperation>& units) 
    {
        Model m("RefineryModel");
        
        // Mock model building for demonstration
        for (const auto& f : feeds) {
            int v = m.add_variable(f.name + "_feed", VariableType::Continuous, {0, f.available_quantity});
            m.objective().add_term(v, -f.purchase_cost);
        }
        
        for (const auto& p : products) {
            int v = m.add_variable(p.name + "_prod", VariableType::Continuous, {p.demand, 1e6});
            m.objective().add_term(v, p.selling_price);
        }
        
        for (const auto& u : units) {
            int v = m.add_variable(u.name + "_unit", VariableType::Continuous, {0, u.capacity});
            m.objective().add_term(v, -u.operating_cost - u.energy_cost);
        }
        
        m.objective().set_sense(OptimizationSense::Maximize);
        return m;
    }
    
    static EconomicAnalysis run_workflow(Model& model) {
        // Step 1: Validation
        // Step 2: Routing
        RoutingResult router_res = SolverRouter::route(model);
        
        // Step 3: Solve (Mocking the exact calculation)
        EconomicAnalysis analysis;
        analysis.revenue = 1500000.0;
        analysis.raw_material_cost = 500000.0;
        analysis.operating_cost = 200000.0;
        analysis.energy_cost = 100000.0;
        analysis.estimated_margin = analysis.revenue - analysis.raw_material_cost - analysis.operating_cost - analysis.energy_cost;
        
        // Expose explicit binding constraints and shadow prices (mocked logic)
        analysis.binding_constraints.push_back({"FeedAvailability_CrudeA", 12.50});
        analysis.binding_constraints.push_back({"UnitCapacity_Distillation", 45.00});
        analysis.binding_constraints.push_back({"ProductQuality_SulfurMax", 1500.00});
        
        return analysis;
    }
};

} // namespace refinery
} // namespace bharatopt
