package K_test;
use lib "../..";
use Test;
use strict;
use warnings;
BEGIN { plan test => 33}

use AI::NeuralNet::Kohonen;
ok(1,1);
use AI::NeuralNet::Kohonen::Node;
ok(1,1);
use AI::NeuralNet::Kohonen::Input;
ok(1,1);

my $net = new AI::NeuralNet::Kohonen;
ok ($net,undef);

$net = new AI::NeuralNet::Kohonen(
	weight_dim => 2,
	input => [
		[1,2,3]
	],
);
ok( ref $net->{input}, 'ARRAY');
ok( $net->{input}->[0]->[0],1);
ok( $net->{input}->[0]->[1],2);
ok( $net->{input}->[0]->[2],3);
ok( $net->{map_dim_a},19);

$net = new AI::NeuralNet::Kohonen(
	weight_dim => 2,
	input => [
		[1,2,3]
	],
	map_dim_x => 10,
	map_dim_y => 20,
);
ok($net->{map_dim_a},15);


# Node test
my $node = new AI::NeuralNet::Kohonen::Node;
ok($node,undef);
$node = new AI::NeuralNet::Kohonen::Node(
	weight => [0.1, 0.6, 0.5],
);
ok( ref $node, 'AI::NeuralNet::Kohonen::Node');
ok( $node->{dim}, 2);
my $input = new AI::NeuralNet::Kohonen::Input(
	dim		=> 2,
	values	=> [1,0,0],
);

ok( sprintf("%.2f",$node->distance_from($input)), 1.19);

$net = AI::NeuralNet::Kohonen->new(
	map_dim_x	=> 14,
	map_dim_y	=> 10,
	epoch_end	=> sub {print"."},
	train_end	=> sub {print"\n"},
	epochs		=> 2,
	table		=>
"3
1 0 0 red
0 1 0 green
0 0 1 blue
",
);
ok( ref $net->{input}, 'ARRAY');
ok( ref $net->{input}->[0],'AI::NeuralNet::Kohonen::Input');
ok( $net->{input}->[0]->{values}->[0],1);
ok( $net->{input}->[0]->{values}->[1],0);
ok( $net->{input}->[0]->{values}->[2],0);
ok( $net->{weight_dim}, 2);
ok( ref $net->{map}, 'ARRAY');
$net->train;
ok( ref $net->{map}, 'ARRAY');
my @bmu = $net->get_results();
ok( ref $bmu[0], 'ARRAY');
ok (ref $net->{map}->[ 0 ]->[ 0 ], "AI::NeuralNet::Kohonen::Node" );

warn "# Class test:\n";
@bmu = $net->get_results([[0.5,0,0]]);
ok (ref $net->{map}->[ $bmu[0]->[1] ]->[ $bmu[0]->[2] ],
	"AI::NeuralNet::Kohonen::Node"
);
# warn $net->{map}->[ $bmu[1] ]->[ $bmu[2] ];#->get_class;
# Get the nearest class?

{
	my $i=0;
	my $targets = [[1, 0, 0],[0,1,0],[0,0,1]];
	my @bmu = $net->get_results($targets);
	# qerror
	my $qerror=0;
	foreach my $j (0..$net->{weight_dim}){ # loop over weights
		$qerror += $targets->[0]->{values}->[$j]
		- $net->{map}->[$bmu[$i]->[1]]->[$bmu[$i]->[2]]->{weight}->[$j];
	}
	ok( $qerror, $net->quantise_error([ [1,0,0] ]));
}

warn "# Input file tests\n";
$net = AI::NeuralNet::Kohonen->new(
	epochs	=> 0,
	input_file => 'ex.dat',
	epoch_end	=> sub {print"."},
	train_end	=> sub {print"\n"},
);
ok( ref $net,'AI::NeuralNet::Kohonen');
ok( ref $net->{input}, 'ARRAY');
ok( scalar @{$net->{input}}, 3840);
ok( $net->{map_dim_x}, 19);
ok ($net->{input}->[$#{$net->{input}}]->{values}->[4], 406.918518);
ok( ref $net->{input}->[$#{$net->{input}}]->{values}, 'ARRAY');
warn "# Training on a big file: this is SLOW, sorry\n";
ok ($net->train,1);
my $filename = substr(time,0,8);
ok ($net->save_file($filename),1);
ok (unlink($filename),1);
__END__

