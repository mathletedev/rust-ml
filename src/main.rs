pub mod lib;

use lib::network::Network;

fn main() {
	let inputs = vec![
		vec![0.0, 0.0],
		vec![0.0, 1.0],
		vec![1.0, 0.0],
		vec![1.0, 1.0],
	];

	let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

	let mut network = Network::new(vec![2, 3, 1], None, None);

	network.train(inputs, outputs, 10000);

	println!("{:?}", network.feed_forward(vec![0.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![0.0, 1.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 1.0]));
}
