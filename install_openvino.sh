# wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18319/l_openvino_toolkit_p_2021.4.752.tgz

export pwd=$PWD

tar -xvzf l_openvino_toolkit_p_2020.4.287.tgz
cd l_openvino_toolkit_p_2020.4.287
sudo ./install.sh

cd /opt/intel/openvino/install_dependencies
sudo -E ./install_openvino_dependencies.sh
echo source /opt/intel/openvino/bin/setupvars.sh >> ~/.bashrc

cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
sudo ./install_prerequisites_onnx.sh

cd $pwd
