package AI::NeuralNet::Kohonen;

use vars qw/$VERSION $TRACE/;
$VERSION = 0.12;	# 13 March 2003
$TRACE = 1;

=head1 NAME

AI::NeuralNet::Kohonen - Kohonen's Self-organising Maps

=cut

use strict;
use warnings;
use Carp qw/croak cluck/;

use AI::NeuralNet::Kohonen::Node;

=head1 SYNOPSIS

	$_ = AI::NeuralNet::Kohonen->new(
		map_dim_x => 39,
		map_dim_y => 19,
		epochs    => 100,
		table     =>
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


=head1 DEPENDENCIES

None

=head1 EXPORTS

None


=head1 CONSTRUCTOR new

Instantiates object fields:

=over 4

=item input_file

A I<SOM_PAK> training file to load. This does not prevent
other input methods (C<input>, C<table>) being processed, but
it does over-ride any specifications (C<weight_dim>) which may
have been explicitly handed to the constructor.

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

=item map_dim_x

=item map_dim_y

The dimensions of the feature map to create - defaults to a toy 19.
(note: this is Perl indexing, starting at zero).

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

=item targeting

If undefined, random targets are chosen; otherwise
they're iterated over. Just for experimental purposes.

=item smoothing

The amount of smoothing to apply by default when C<smooth>
is applied (see L</METHOD smooth>).

=back

Private fields:

=over 4

=item time_constant

The number of iterations (epochs) to be completed, over the log of the map radius.

=item t

The current epoch, or moment in time.

=item l

The current learning rate.

=item map_dim_a

Average of the map dimensions.

=back

=cut

sub new {
	my $class			= shift;
	my %args			= @_;
	my $self 			= bless \%args,$class;
	$self->_process_table if defined $self->{table};	# Creates {input}
	$self->load_file($self->{input_file}) if defined $self->{input_file};	# Creates {input}
	if (not defined $self->{input}){
		cluck "No {input} supplied!";
		return undef;
	}
	# Error check this, not ignore
	if (not $self->{input_file}){
		$self->{weight_dim}		= $#{$self->{input}->[0]};
	}
	$self->{map_dim_x}		= 19 unless defined $self->{map_dim_x};
	$self->{map_dim_y}		= 19 unless defined $self->{map_dim_y};
	# Legacy from...yesterday
	if ($self->{map_dim}){
		$self->{map_dim_x} = $self->{map_dim_y} = $self->{map_dim}
	}
	if ($self->{map_dim_x}>$self->{map_dim_y}){
		$self->{map_dim_a} = $self->{map_dim_y} + (($self->{map_dim_x}-$self->{map_dim_y})/2)
	} else {
		$self->{map_dim_a} = $self->{map_dim_x} + (($self->{map_dim_y}-$self->{map_dim_x})/2)
	}
	$self->{epochs}			= 99 unless defined $self->{epochs};
	$self->{time_constant}	= $self->{epochs} / log($self->{map_dim_a});	# to base 10?
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
	croak "{map_dim_x} not set" unless $self->{map_dim_x};
	croak "{map_dim_y} not set" unless $self->{map_dim_y};
	for my $x (0..$self->{map_dim_x}){
		$self->{map}->[$x] = [];
		for my $y (0..$self->{map_dim_y}){
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
		my $bmu = $self->find_bmu($target);
		$self->_adjust_neighbours_of($bmu,$target);
		$self->_decay_learning_rate;
		&{$self->{epoch_end}} if exists $self->{epoch_end};
	}
	&{$self->{train_end}} if $self->{train_end};
	return 1;
}


=head1 METHOD find_bmu

Find the Best Matching Unit in the map and return the x/y index.

Accepts: a reference to an array that is the target.

Returns: a reference to an array that is the BMU (and should
perhaps be abstracted as an object in its own right), indexed as follows:

=over 4

=item 0

euclidean distance from the supplied target

=item 1

I<x> co-ordinate in the map

=item 2

I<y> co-ordinate in the map

=back

See L</METHOD get_weight_at>,
and L<AI::NeuralNet::Kohonen::Node/distance_from>,

=cut


sub find_bmu { my ($self,$target) = (shift,shift);
	my $closest = [];	# [value, x,y] value and co-ords of closest match
	for my $x (0..$self->{map_dim_x}){
		for my $y (0..$self->{map_dim_y}){
			my $distance = $self->{map}->[$x]->[$y]->distance_from( $target );
			$closest = [$distance,0,0] if $x==0 and $y==0;
			$closest = [$distance,$x,$y] if $distance < $closest->[0];
		}
	}
	return $closest;
}

=head1 METHOD get_weight_at

Returns a reference to the weight array at the supplied I<x>,I<y>
co-ordinates.

Accepts: I<x>,I<y> co-ordinates, each a scalar.

Returns: reference to an array that is the weight of the node, or
C<undef> on failure.

=cut

sub get_weight_at { my ($self,$x,$y) = (shift,shift,shift);
	return undef if $x<0 or $y<0 or $x>$self->{map_dim_x} or $y>$self->{map_dim_y};
	return $self->{map}->[$x]->[$y]->{weight};
}

=head1 PRIVATE METHOD find_bmu

Depreciated - should have been public to begin with.

=cut

sub _find_bmu { return find_bmu(@_) }


=head1 METHOD get_results

Finds and returns the results for all input vectors (C<input>),
placing the values in the array reference that is the C<results>
field, and, depending on calling context, returning it either as
an array or as it is.

Individual results are in the array format as described in
L<METHOD find_bmu>.

See L<METHOD find_bmu>, and L</METHOD get_weight_at>.


=cut

sub get_results { my $self=shift;
	$self->{results} = [];
	foreach my $target (@{ $self->{input} }){
		push @{$self->{results}}, $self->find_bmu($target);
	}
	return wantarray? @{$self->{results}} : $self->{results};
}


=head1 METHOD dump

Print the current weight values to the screen.

=cut

sub dump { my $self=shift;
	print "    ";
	for my $x (0..$self->{map_dim_x}){
		printf ("  %02d ",$x);
	}
	print"\n","-"x107,"\n";
	for my $x (0..$self->{map_dim_x}){
		for my $w (0..$self->{weight_dim}){
			printf ("%02d | ",$x);
			for my $y (0..$self->{map_dim_y}){
				printf("%.2f ", $self->{map}->[$x]->[$y]->{weight}->[$w]);
			}
			print "\n";
		}
		print "\n";
	}
}

=head1 METHOD smooth

Perform gaussian smoothing upon the map.

Accepts: the length of the side of the square gaussian mask to apply.
If not supplied, uses the value in the field C<smoothing>; if that is
empty, uses the square root of the average of the map dimensions
(C<map_dim_a>).

Returns: a true value.

=cut

sub smooth { my ($self,$smooth) = (shift,shift);
	$smooth = $self->{smoothing} if not $smooth and defined $self->{smoothing};
	$smooth = int( sqrt $self->{map_dim_a} );
	my $mask = _make_gaussian_mask($smooth);

	# For every weight at every point
	for my $x (0..$self->{map_dim_x}){
		for my $y (0..$self->{map_dim_y}){
			for my $w (0..$self->{weight_dim}){
				# Apply the mask
				for my $mx (0..$smooth){
					for my $my (0..$smooth){
						$self->{map}->[$x]->[$y]->{weight}->[$w] *= $mask->[$mx]->[$my];
					}
				}
			}
		}
	}
	return 1;
}


=head1 METHOD tk_dump;

Extended and moved to the package C<AI::NeuralNet::Kohonen::Demo::RGB>.

=cut


=head1 PRIVATE METHOD _select_target

Return a random target from the training set in the C<input> field,
unless the C<targeting> field is defined, when the targets are
iterated over.

=cut

sub _select_target { my $self=shift;
	if (not $self->{targeting}){
		return $self->{input}->[
			(int rand(scalar @{$self->{input}}))
		];
	}
	else {
		$self->{tar}++;
		if ($self->{tar}>=$#{ $self->{input} }){
			$self->{tar} = 0;
		}
		return $self->{input}->[$self->{tar}];
	}
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
		($self->{map_dim_a}/2) * exp(- $self->{t} / $self->{time_constant})
	);

	# Distance from co-ord vector (0,0) as integer
	# Basically map_width * y  +  x
	my $centre = ($self->{map_dim_a}*$bmu->[2])+$bmu->[1];

	for my $x ($bmu->[1]-$neighbour_radius .. $bmu->[1]+$neighbour_radius){
		next if $x<0 or $x>$self->{map_dim_x};		# Ignore those not mappable
		for my $y ($bmu->[2]-$neighbour_radius .. $bmu->[2]+$neighbour_radius){
			next if $y<0 or $y>$self->{map_dim_y};	# Ignore those not mappable
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


=head1 PRIVATE FUNCTION _make_gaussian_mask

Accepts: size of mask.

Returns: reference to a 2d array that is the mask.

=cut

sub _make_gaussian_mask { my ($smooth) = (shift);
	my $f = 4; # Cut-off threshold
	my $g_mask_2d = [];
	for my $x (0..$smooth){
		$g_mask_2d->[$x] = [];
		for my $y (0..$smooth){
			$g_mask_2d->[$x]->[$y] =
				_gauss_weight( $x-($smooth/2), $smooth/$f)
			  * _gauss_weight( $y-($smooth/2), $smooth/$f );
		}
	}
	return $g_mask_2d;
}

=head1 PRIVATE FUNCTION _gauss_weight

Accepts: two paramters: the first, C<r>, gives the distance from the mask centre,
the second, C<sigma>, specifies the width of the mask.

Returns the gaussian weight.

See also L<_decay_learning_rate>.

=cut

sub _gauss_weight { my ($r, $sigma) = (shift,shift);
	return exp( -($r**2) / (2 * $sigma**2) );
}


=head1 FILE FORMAT

I<SOM_PAK> file format version 3.1 (April 7, 1995),
Helsinki University of Technology, Espoo:

=over 4

The input data is stored in ASCII-form as a list of entries, one line
...for each vectorial sample.

The first line of the file is reserved for status knowledge of the
entries; in the present version it is used to define the following
items (these items MUST occur in the indicated order):

   - Dimensionality of the vectors (integer, compulsory).
   - Topology type, either hexa or rect (string, optional, case-sensitive).
   - Map dimension in x-direction (integer, optional).
   - Map dimension in y-direction (integer, optional).
   - Neighborhood type, either bubble or gaussian (string, optional, case-sen-
      sitive).

...

Subsequent lines consist of n floating-point numbers followed by an
optional class label (that can be any string) and two optional
qualifiers (see below) that determine the usage of the corresponding
data entry in training programs.  The data files can also contain an
arbitrary number of comment lines that begin with '#', and are
ignored. (One '#' for each comment line is needed.)

If some components of some data vectors are missing (due to data
collection failures or any other reason) those components should be
marked with 'x'...[in processing, these] are ignored.

=back

Not implimented (yet):

	I<neighbourhood type>, which is always gaussian.
	i<x> for missing data.
	class labels
	the two optional qualifiers

=back

Requires: a path to a file.

Returns C<undef> on failure.

=cut

sub load_file { my ($self,$path) = (shift,shift);
	local *IN;
	if (not open IN,$path){
		warn "Could not open file <$path>: $!";
		return undef;
	}
	@_ = <IN>;
	close IN;
	chomp @_;
	my @specs = split/\s+/,(shift @_);
	#- Dimensionality of the vectors (integer, compulsory).
	$self->{weight_dim} = shift @specs;
	$self->{weight_dim}--; # Perl indexing
	#- Topology type, either hexa or rect (string, optional, case-sensitive).
	$self->{display}    = shift @specs;
	if (not defined $self->{display}){

	} elsif ($self->{display} eq 'hexa'){
		$self->{display} = 'hex'
	} elsif ($self->{display} eq 'rect'){
		$self->{display} = undef;
	} else {
		warn "Unknown display option '$self->{display}' in file <$path> - defaulting.";
		$self->{display} = undef;
	}
	#- Map dimension in x-direction (integer, optional).
	$self->{map_dim_x}    = shift @specs;
	#- Map dimension in y-direction (integer, optional).
	$self->{map_dim_y}    = shift @specs;
	#- Neighborhood type, either bubble or gaussian (string, optional, case-sen- sitive).
	# not implimented

	# Format input data
	foreach (@_){
		s/#.*$//g;
		next if /^$/;
		my @i = split /\s+/;
		push @{$self->{input}}, \@i;
	}
	return 1;
}

=head1 METHOD save_file

Saves the map file in I<SOM_PAK> format (see L<METHOD load_file>)
at the path specified in the first argument.

Return C<undef> on failure, a true value on success.

=cut

sub save_file { my ($self,$path) = (shift,shift);
	local *OUT;
	if (not open OUT,">$path"){
		warn "Could not open file for writing <$path>: $!";
		return undef;
	}
	#- Dimensionality of the vectors (integer, compulsory).
	print OUT ($self->{weight_dim}+1)," ";	# Perl indexing
	#- Topology type, either hexa or rect (string, optional, case-sensitive).
	if (not defined $self->{display}){
		print OUT "rect ";
	} else { # $self->{display} eq 'hex'
		print OUT "hexa ";
	}
	#- Map dimension in x-direction (integer, optional).
	print OUT $self->{map_dim_x}." ";
	#- Map dimension in y-direction (integer, optional).
	print OUT $self->{map_dim_y}." ";
	#- Neighborhood type, either bubble or gaussian (string, optional, case-sen- sitive).
	print OUT "gaussian ";
	# End of header
	print OUT "\n";

	# Format input data
	foreach (@{$self->{input}}){
		print OUT join("\t",@$_),"\n";
	}
	# EOF
	print OUT chr 26;
	close OUT;
	return 1;
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
