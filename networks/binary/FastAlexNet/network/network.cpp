#include "network.hpp"


Network::Network(void) :
	/* FastAlexNet */
	conv1("params\\conv1_W.npy", "params\\conv1_b.npy",
		"params\\decomposition\\conv1\\basis_6\\M.npy", "params\\decomposition\\conv1\\basis_6\\c.npy"),
	conv2("params\\conv2_W.npy", "params\\conv2_b.npy",
		"params\\decomposition\\conv2\\basis_6\\M.npy", "params\\decomposition\\conv2\\basis_6\\c.npy"),
	conv3("params\\conv3_W.npy", "params\\conv3_b.npy",
		"params\\decomposition\\conv3\\basis_6\\M.npy", "params\\decomposition\\conv3\\basis_6\\c.npy"),
	conv4("params\\conv4_W.npy", "params\\conv4_b.npy",
		"params\\decomposition\\conv4\\basis_6\\M.npy", "params\\decomposition\\conv4\\basis_6\\c.npy"),
	conv5("params\\conv5_W.npy", "params\\conv5_b.npy",
		"params\\decomposition\\conv5\\basis_6\\M.npy", "params\\decomposition\\conv5\\basis_6\\c.npy"),
	fc6("params\\fc6_W.npy", "params\\fc6_b.npy",
		"params\\decomposition\\fc6\\basis_6\\M.npy", "params\\decomposition\\fc6\\basis_6\\c.npy"),
	fc7("params\\fc7_W.npy", "params\\fc7_b.npy",
		"params\\decomposition\\fc7\\basis_6\\M.npy", "params\\decomposition\\fc7\\basis_6\\c.npy"),
	fc8("params\\fc8_W.npy", "params\\fc8_b.npy",
		"params\\decomposition\\fc8\\basis_6\\M.npy", "params\\decomposition\\fc8\\basis_6\\c.npy"),
	lrn1(5, 2, 0.0001, 0.75),
	lrn2(5, 2, 0.0001, 0.75)

	/* AlexNet */
	/*conv1("params\\conv1_W.npy", "params\\conv1_b.npy"),
	conv2("params\\conv2_W.npy", "params\\conv2_b.npy"),
	conv3("params\\conv3_W.npy", "params\\conv3_b.npy"),
	conv4("params\\conv4_W.npy", "params\\conv4_b.npy"),
	conv5("params\\conv5_W.npy", "params\\conv5_b.npy"),
	fc6("params\\fc6_W.npy", "params\\fc6_b.npy"),
	fc7("params\\fc7_W.npy", "params\\fc7_b.npy"),
	fc8("params\\fc8_W.npy", "params\\fc8_b.npy"),
	lrn1(5, 2, 0.0001, 0.75),
	lrn2(5, 2, 0.0001, 0.75)*/
{}