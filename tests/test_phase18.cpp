#include <gtest/gtest.h>
#include "bharatopt/parser/mps_parser.hpp"

using namespace bharatopt;

TEST(Phase18Test, ParseMPSFile) {
    // We expect the file to be present at this relative path when run from build/tests
    std::string filepath = "../../data/test_model.mps";
    
    Model model = MPSParser::parse(filepath);
    
    // Model should have 2 variables (X, Y) and 2 constraints (C1, C2)
    EXPECT_EQ(model.variables().size(), 2);
    EXPECT_EQ(model.constraints().size(), 2);
    
    // Check objective terms (X -> 3.0, Y -> 2.0)
    auto obj_terms = model.objective().linear_terms();
    EXPECT_EQ(obj_terms.size(), 2);
    
    // Verifying terms exist
    bool found_x_obj = false;
    for(auto& term : obj_terms) {
        if(term.second == 3.0) found_x_obj = true;
    }
    EXPECT_TRUE(found_x_obj);
}
