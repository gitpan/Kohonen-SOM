package K_test;
use lib "../..";
use Test;
BEGIN { plan test => 19}

use AI::NeuralNet::Kohonen;
ok(1,1);
use AI::NeuralNet::Kohonen::Node;
ok(1,1);

$_ = new AI::NeuralNet::Kohonen;
ok ($_,undef);

$_ = new AI::NeuralNet::Kohonen(
	input => [
		[1,2,3]
	],
);
ok( ref $_->{input}, 'ARRAY');
ok( $_->{input}->[0]->[0],1);
ok( $_->{input}->[0]->[1],2);
ok( $_->{input}->[0]->[2],3);
ok($_->{map_dim_a},19);

$_ = new AI::NeuralNet::Kohonen(
	input => [
		[1,2,3]
	],
	map_dim_x => 10,
	map_dim_y => 20,
);
ok($_->{map_dim_a},15);


# Node test
my $node = new AI::NeuralNet::Kohonen::Node;
ok($node,undef);
$node = new AI::NeuralNet::Kohonen::Node(
	weight => [0.1, 0.6, 0.5],
);
ok( ref $node, 'AI::NeuralNet::Kohonen::Node');
ok( $node->{dim}, 2);
ok( sprintf("%.2f",$node->distance_from([1,0,0])), 1.19);

$_ = AI::NeuralNet::Kohonen->new(
	map_dim	=> 39,
	epochs => 10,
	table=>
"R G B
1 0 0
0 1 0
0 0 1
",
);
ok( ref $_->{input}, 'ARRAY');
ok( $_->{input}->[0]->[0],1);
ok( $_->{input}->[0]->[1],0);
ok( $_->{input}->[0]->[2],0);
ok( $_->{weight_dim}, 2);

#$_->dump;
$_->train;
#warn "Trained.\n";
@_ = $_->get_results;
foreach (@_){
	print join",",@$_,"\n";
}

ok(1,1);

__END__

1 1 0
1 0 1
0 1 1
1 1 1
