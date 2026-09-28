include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_engine 
    test_engine.cpp 
    test_phase1.cpp 
    test_phase2.cpp
    test_phase3_4_5.cpp
    test_phase6.cpp
    test_phase7_8_9.cpp
    test_phase10_11_12.cpp
    test_phase13_14.cpp
    test_phase16_17.cpp
    test_phase18.cpp
)
target_link_libraries(test_engine PRIVATE engine_lib gtest_main)

add_test(NAME EngineTest COMMAND test_engine)
