add_executable(demo_corr_fpfh corr_fpfh.cpp)
target_link_libraries(demo_corr_fpfh PRIVATE turboreg ${TORCH_LIBRARIES})
