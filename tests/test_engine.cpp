#include <gtest/gtest.h>
#include "bharatopt/engine.hpp"

TEST(EngineTest, VersionCheck) {
    bharatopt::Engine engine;
    EXPECT_EQ(engine.get_version(), 1);
}
