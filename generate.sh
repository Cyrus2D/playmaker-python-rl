python3 -m grpc_tools.protoc -I./protos --python_out=./sample-low-level-team --pyi_out=./sample-low-level-team --grpc_python_out=./sample-low-level-team ./protos/*.proto
python3 -m grpc_tools.protoc -I./protos --python_out=./sample-high-level-team --pyi_out=./sample-high-level-team --grpc_python_out=./sample-high-level-team ./protos/*.proto