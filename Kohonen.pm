package AI::NeuralNet::Kohonen;

use vars qw/$VERSION $TRACE/;
$VERSION = 0.1;
$TRACE = 1;

=head1 NAME

AI::NeuralNet::Kohonen - Kohonen's Self-organising Maps

=cut

use strict;
use warnings;
use Carp qw/croak cluck carp confess/;

use AI::NeuralNet::Kohonen::Node;

=head1 SYNOPSIS

	$_ = AI::NeuralNet::Kohonen->new(
		map_dim	=> 39,
		epochs => 100,
		table=>
	"R G B
	1 0 0
	0 1 0
	0 0 1
	1 1 0
	1 0 1
	0 1 1
	1 1 1
	");

	$_->dump;
	$_->tk_dump;

	$_->train;

	$_->dump;
	$_->tk_dump;
	exit;

=head1 DESCRIPTION

An illustrative implimentation of Kohonen's Self-organising Feature Maps (SOMs)
in Perl.

It's not fast - it's illustrative.

In fact, it's slow: but it's illustrative....

Have a look at L<AI::NeuralNet::Kohonen::Demo::RGB>.

I'll add some more text here later.

=head1 CONSTRUCTOR new

Sets up object fields:

=over 4

=item input

A reference to an array of training vectors, within which each vector
is represented by an array:

	[ [v1a, v1b, v1c], [v2a,v2b,v2c], ..., [vNa,vNb,vNc] ]

See also C<table>.

=item table

A scalar that is a table, lines delimited by
C<[\r\f\n]+>, columns by whitespace, initial whitespace stripped.
First line should be column names, the following lines should be just data.
See also C<input>.

=item input_names

A name for each dimension of the input vectors.

=item map_dim

The size of one dimension of the feature map to create - defaults to a toy 19.
(note: this is Perl indexing).

=item epochs

Number of epochs to run for (see L<METHOD train>).

=item learning_rate

The initial learning rate.

=item train_start

Reference to code to call at the begining of training.

=item epoch_end

Reference to code to call at the end of every epoch
(such as a display routine).

=item train_end

Reference to code to call at the end of training.

=back

Private fields:

=over 4

=item time_constant

The number of iterations (epochs) to be completed, over the log of the map radius.

=item t

The current epoch, or moment in time.

=item l

The current learning rate.

=back

=cut

sub new {
	my $class			= shift;
	my %args			= @_;
	my $self 			= bless \%args,$class;
	$self->_process_table if defined $self->{table};	# Creates {input}
	if (not defined $self->{input}){
		cluck "No {input} supplied!";
		return undef;
	}
	$self->{weight_dim}		= $#{$self->{input}->[0]};
	$self->{map_dim}		= 19 unless defined $self->{map_dim};
	$self->{epochs}			= 99 unless defined $self->{epochs};
	$self->{time_constant}	= $self->{epochs} / log($self->{map_dim});	# to base 10?
	$self->{learning_rate}	= 0.5 unless $self->{learning_rate};
	$self->{l}				= $self->{learning_rate};
	$self->_create_map;
	return $self;
}


#
# Processes the 'table' paramter to the constructor
#
sub _process_table { my $self = shift;
	die "Accepts just a scalar" if $#_>0 or ref $_[0];
	my @input;
	my ($input,@table) = split /[\n\r\f]+/,$self->{table};
	undef $self->{table};
	@{$self->{input_names}} = split /\s+/,$input;
	while (my $i = shift @table){
		my @i = split /\s+/,$i;
		push @{$self->{input}}, \@i;
	}
}


#
# PRIVATE METHOD _create_map
#
# Populates the map with nodes that contain random nubmers.
#
sub _create_map { my $self=shift;
	croak "{weight_dim} not set" unless $self->{weight_dim};
	croak "{map_dim} not set" unless $self->{map_dim};
	for my $x (0..$self->{map_dim}){
		$self->{map}->[$x] = [];
		for my $y (0..$self->{map_dim}){
			$self->{map}->[$x]->[$y] = new AI::NeuralNet::Kohonen::Node(
				dim => $self->{weight_dim}
			);
		}
	}
}


=head1 METHOD train

Optionally accepts a parameter that is the number of epochs
to train for - default is the value in the C<epochs> field.

For every C<epoch>, iterates:

	- selects a random target from the input array;
	- finds the best bmu
	- adjusts neighbours of the bmu
	- decays the learning rate

=cut

sub train { my ($self,$epochs) = (shift,shift);
	$epochs = $self->{epochs} unless defined $epochs;
	&{$self->{train_start}} if exists $self->{train_start};
	for my $epoch (0..$epochs){
		$self->{t} = $epoch;
		my $target = $self->_select_target;
		my $bmu = $self->_find_bmu($target);
		$self->_adjust_neighbours_of($bmu,$target);
		$self->_decay_learning_rate;
		&{$self->{epoch_end}} if exists $self->{epoch_end};
	}
	&{$self->{train_end}} if $self->{train_end};
	return 1;
}


=head1 METHOD dump

Print the current weight values to the screen.

=cut

sub dump { my $self=shift;
	print "    ";
	for my $x (0..$self->{map_dim}){
		printf ("  %02d ",$x);
	}
	print"\n","-"x107,"\n";
	for my $x (0..$self->{map_dim}){
		for my $w (0..$self->{weight_dim}){
			printf ("%02d | ",$x);
			for my $y (0..$self->{map_dim}){
				printf("%.2f ", $self->{map}->[$x]->[$y]->{weight}->[$w]);
			}
			print "\n";
		}
		print "\n";
	}
}


=head1 METHOD tk_dump;

Dumps the weights to a TK screen.

=cut

sub tk_dump { my $self=shift;
	$self->{display_scale} = 10;
	my $size = $self->{map_dim} * $self->{display_scale};
	eval "
		use Tk;
		use Tk::Canvas;
	";
	my $mw = MainWindow->new(
		-width	=> ($size+200),
		-height	=> ($size+200),
	);
	my $c = $mw->Canvas(
		-width	=> $size+100,
		-height	=> $size+100,
	);

	for my $x (0..$self->{map_dim}){
		for my $y (0..$self->{map_dim}){
			my $colour = sprintf("#%02x%02x%02x",
				(int (255 * $self->{map}->[$x]->[$y]->{weight}->[0])),
				(int (255 * $self->{map}->[$x]->[$y]->{weight}->[1])),
				(int (255 * $self->{map}->[$x]->[$y]->{weight}->[2])),
			);
			$c->create(
				rectangle	=> [
					(1+$x)*$self->{display_scale} ,
					(1+$y)*$self->{display_scale} ,
					(1+$x)*($self->{display_scale})+$self->{display_scale} ,
					(1+$y)*($self->{display_scale})+$self->{display_scale}
				],
				-outline	=> "black",
				-fill 		=> $colour,
			);
		}
	}
	$c->pack();
	eval "
		MainLoop;
	";
	return 1;
}


=head1 PRIVATE METHOD _select_target

Return a random target from the training set in the C<input> field.

=cut

sub _select_target { my $self=shift;
	return $self->{input}->[
		(int rand(scalar @{$self->{input}}))
	];
}


=head1 PRIVATE METHOD _find_bmu

Find the Best Matching Unit in the map and return the x/y index.

Accepts: a reference to an array that is the target.

Returns: a reference to an array that is the BMU, of the format

	[euclidean-distance-from-target, our-x-co-ord, our-y-co-ord]

See L<AI::NeuralNet::Kohonen::Node/distance_from>.

=cut

sub _find_bmu { my ($self,$target) = (shift,shift);
	my $closest = [];	# [value, x,y] value and co-ords of closest match
	for my $x (0..$self->{map_dim}){
		for my $y (0..$self->{map_dim}){
			my $distance = $self->{map}->[$x]->[$y]->distance_from( $target );
			$closest = [$distance,0,0] if $x==0 and $y==0;
			$closest = [$distance,$x,$y] if $distance < $closest->[0];
		}
	}
	return $closest;
}


=head1 PRIVATE METHOD _adjust_neighbours_of

Accepts: a reference to an array containing
the distance of the BMU from the target, and
the x and y co-ordinates of the BMU in the map;
a reference to an array that is the target.

Returns: true.

=head2 FINDING THE NEIGHBOURS OF THE BMU

	                        (      t   )
	sigma(t) = sigma(0) exp ( - ------ )
	                        (   lambda )

Where C<sigma> is the width of the map at any stage
in time (C<t>), and C<lambda> is a time constant.

Lambda is our field C<time_constant>.

The map radius is naturally just half the map width.

=head2 ADJUSTING THE NEIGHBOURS OF THE BMU

	W(t+1) = W(t) + THETA(t) L(t)( V(t)-W(t) )

Where C<L> is the learning rate, C<V> the target vector,
and C<W> the weight. THETA(t) represents the influence
of distance from the BMU upon a node's learning, and
is calculated by the C<Node> class - see
L<AI::NeuralNet::Kohonen::Node/distance_effect>.

=cut

sub _adjust_neighbours_of { my ($self,$bmu,$target) = (shift,shift,shift);
	my $neighbour_radius = int (
		($self->{map_dim}/2) * exp(- $self->{t} / $self->{time_constant})
	);
	# Distance from co-ord vector (0,0) as integer
	# Basically map_width * y  +  x
	my $centre = ($self->{map_dim}*$bmu->[2])+$bmu->[1];

	for my $x ($bmu->[1]-$neighbour_radius .. $bmu->[1]+$neighbour_radius){
		next if $x<0 or $x>$self->{map_dim};		# Ignore those not mappable
		for my $y ($bmu->[2]-$neighbour_radius .. $bmu->[2]+$neighbour_radius){
			next if $y<0 or $y>$self->{map_dim};	# Ignore those not mappable
			# Skip node if it is out of the circle of influence
			next if (
				(($bmu->[1] - $x) * ($bmu->[1] - $x)) + (($bmu->[2] - $y) * ($bmu->[2] - $y))
			) > ($neighbour_radius * $neighbour_radius);

			# Adjust the weight
			for my $w (0..$self->{weight_dim}){
				my $weight = \$self->{map}->[$x]->[$y]->{weight}->[$w];
				$$weight = $$weight + (
					$self->{map}->[$x]->[$y]->distance_effect($bmu->[0], $neighbour_radius)
					* ( $self->{l} * ($target->[$w] - $$weight) )
				);
			}
		}
	}
}


# Poor: selects a square around the bmu
sub SQUARE_adjust_neighbours_of { my ($self,$bmu,$target) = (shift,shift,shift);
	my $neighbour_radius = int (
		($self->{map_dim}/2) * exp(- $self->{t} / $self->{time_constant})
	);
	for my $x ($bmu->[1]-$neighbour_radius .. $bmu->[1]+$neighbour_radius){
		next if $x<0 or $x>$self->{map_dim};
		for my $y ($bmu->[2]-$neighbour_radius .. $bmu->[2]+$neighbour_radius){
			next if $y<0 or $y>$self->{map_dim};
			# Adjust the weight
			for my $w (0..$self->{weight_dim}){
				my $weight = \$self->{map}->[$x]->[$y]->{weight}->[$w];
				$$weight = $$weight + (
					$self->{map}->[$x]->[$y]->distance_effect($bmu->[0], $neighbour_radius)
					* ( $self->{l} * ($target->[$w] - $$weight) )
				);
			}
		}
	}
}

=head1 PRIVATE METHOD _decay_learning_rate

Performs a gaussian decay upon the learning rate (our C<l> field).

	              (       t   )
	L(t) = L  exp ( -  ------ )
	        0     (    lambda )

=cut

sub _decay_learning_rate { my $self=shift;
	$self->{l} =  (
		$self->{learning_rate} * exp(- $self->{t} / $self->{time_constant})
	);
}


__END__
1;


=head1 SEE ALSO

See L<AI::NeuralNet::Kohonen::Node/distance_from>;
L<AI::NeuralNet::Kohonen::Demo::RGB>.

A very nice explanation of Kohonen's algorithm:
L<AI-Junkie SOM tutorial part 1|http://www.fup.btinternet.co.uk/aijunkie/som1.html>

=head1 AUTHOR AND COYRIGHT

This implimentation Copyright (C) Lee Goddard, 2003.
All Rights Reserved.

Available under the same terms as Perl itself.
