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

$_ = new AI::NeuralNet::Kohonen;
ok ($_,undef);

$_ = new AI::NeuralNet::Kohonen(
	weight_dim => 2,
	input => [
		[1,2,3]
	],
);
ok( ref $_->{input}, 'ARRAY');
ok( $_->{input}->[0]->[0],1);
ok( $_->{input}->[0]->[1],2);
ok( $_->{input}->[0]->[2],3);
ok( $_->{map_dim_a},19);

$_ = new AI::NeuralNet::Kohonen(
	weight_dim => 2,
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
my $input = new AI::NeuralNet::Kohonen::Input(
	dim		=> 2,
	values	=> [1,0,0],
);

ok( sprintf("%.2f",$node->distance_from($input)), 1.19);

$_ = AI::NeuralNet::Kohonen->new(
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
ok( ref $_->{input}, 'ARRAY');
ok( ref $_->{input}->[0],'AI::NeuralNet::Kohonen::Input');
ok( $_->{input}->[0]->{values}->[0],1);
ok( $_->{input}->[0]->{values}->[1],0);
ok( $_->{input}->[0]->{values}->[2],0);
ok( $_->{weight_dim}, 2);

#$_->dump;
$_->train;

@_ = $_->get_results;
foreach my $i (@_){
#	print "Result (distance, x,y): ", join(",",@$i),"\n";
#	print "Class: ",$_->{map}->[$i->[1]]->[$i->[2]]->{class},"\n";
#	print "Weights: ",join(",",@{$_->{map}->[$i->[1]]->[$i->[2]]->{weight}}),"\n";
}
# We should have classified our known points correctly
ok ($_->{map}->[$_[0]->[1]]->[$_[0]->[2]]->{class},'red');
ok ($_->{map}->[$_[1]->[1]]->[$_[1]->[2]]->{class},'green');
ok ($_->{map}->[$_[2]->[1]]->[$_[2]->[2]]->{class},'blue');

# Quantise error
{
	my $i=0;
	my $targets = [[1, 0, 0]];
	my @bmu = $_->get_results($targets);
	# qerror
	my $qerror=0;
	foreach my $j (0..$_->{weight_dim}){ # loop over weights
		$qerror += $targets->[0]->{values}->[$j]
		- $_->{map}->[$bmu[$i]->[1]]->[$bmu[$i]->[2]]->{weight}->[$j];
	}
	ok( $qerror, $_->quantise_error([ [1,0,0] ]));
}

{
	my $i=0;
	my $targets = [[0.5, 0, 0]];
	my @bmu = $_->get_results($targets);
	# qerror
	my $qerror=0;
	foreach my $j (0..$_->{weight_dim}){ # loop over weights
		$qerror += $targets->[0]->{values}->[$j]
		- $_->{map}->[$bmu[$i]->[1]]->[$bmu[$i]->[2]]->{weight}->[$j];
	}
	ok( $qerror, $_->quantise_error([ [0.5,0,0] ]));
}


warn "Class:\n";
my @bmu = $_->get_results([[0.5,0,0]]);
warn $_->{map}->[$bmu[0]->[1]]->[$bmu[0]->[2]]->{class};
# Get the nearest class?

{
	my $i=0;
	my $targets = [[1, 0, 0],[0,1,0],[0,0,1]];
	my @bmu = $_->get_results($targets);
	# qerror
	my $qerror=0;
	foreach my $j (0..$_->{weight_dim}){ # loop over weights
		$qerror += $targets->[0]->{values}->[$j]
		- $_->{map}->[$bmu[$i]->[1]]->[$bmu[$i]->[2]]->{weight}->[$j];
	}
	ok( $qerror, $_->quantise_error([ [1,0,0] ]));
}


$_ = AI::NeuralNet::Kohonen->new(
	epochs	=> 1,
	input_file => 'ex.dat',
	epoch_end	=> sub {print"."},
	train_end	=> sub {print"\n"},
);
ok( ref $_,'AI::NeuralNet::Kohonen');
ok( ref $_->{input}, 'ARRAY');
ok( scalar @{$_->{input}}, 3840);
ok( $_->{map_dim_x}, 19);
ok ($_->{input}->[$#{$_->{input}}]->{values}->[4], 406.918518);
ok ($_->train,1);
my $filename = substr(time,0,8);
ok ($_->save_file($filename),1);
ok (unlink($filename),1);
__END__

