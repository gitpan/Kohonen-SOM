package AI::NeuralNet::Kohonen::Node;

use vars qw/$VERSION $TRACE/;
$VERSION = 0.1;
$TRACE = 1;

=head1 NAME

AI::NeuralNet::Kohonen::Node - a node for AI::NeuralNet::Kohonen

=head1 DESCRIPTION

Implimentation of a node in a SOM - see
L<AI::NeuralNet::Kohonen>.

=cut

use strict;
use warnings;
use Carp qw/cluck carp confess croak/;

=head1 CONSTRUCTOR (new)

=over 4

=item dim

The number of dimensions of this node's weights.
Do not supply if you are supplying C<weight>.

=item weight

Optional: a reference to an array containing the
weight for this node. Supplying this allows the
constructor to work out C<dim>, above.

=back

=cut

sub new {
	my $class	= shift;
	my %args	= @_;
	my $self 	= bless \%args,$class;
	if (not defined $self->{weight}){
		if (not defined $self->{dim}){
			cluck "No {dim} or {weight}!";
			return undef;
		}
		$self->{weight} = [];
		for my $w (0..$self->{dim}){
			$self->{weight}->[$w] = rand;
		}
	} elsif (not ref $self->{weight} or ref $self->{weight} ne 'ARRAY') {
		warn "{weight} should be an array reference!";
		return undef;
	} else {
		$self->{dim} = $#{$self->{weight}};
	}
	return $self;
}


=head1 METHOD distance_from

Find the distance of this node from the target.

Accepts: the target vector as an array reference.

Returns: the distance.

	               __________________
	              / i=n            2
	Distance  =  /   E  ( V  -  W )
	           \/   i=0    i     i

Where C<V> is the current input vector, and
C<W> is this node's weight vector.

=cut

sub distance_from { my ($self) = (shift);
	if (not $_[0] or not ref $_[0] or ref $_[0] ne 'ARRAY'){
		croak "distance_from requires a target array!";
	}
	my @target = @{$_[0]};
	if ($#target != $self->{dim}){
		croak "distance_from requires its target array dim match {dim}!\n"
		."(".($#target)." v {".$self->{dim}."} ) ";
	}
	my $distance = 0;
	for (my $i=0; $i<=$self->{dim}; ++$i){
		$distance += (($target[$i]-$self->{weight}->[$i]) * ($target[$i]-$self->{weight}->[$i]));
	}
	return sqrt($distance);
}


=head1 METHOD distance_effect

Calculates the effect on learning of distance from a given point
(intended to be the BMU).

Accepts:
the distance of this node from the given point;
the radius of the neighbourhood of affect around the given point.

Returns:

	               (            2  )
	               (    distance   )
	THETA(t) = exp ( - ----------- )
	               (          2    )
	               (   2 sigma (t) )

Where C<distance> is the distance of the node from the BMU
(that is, C<$bmu-E<gt>[0]> in the method's paramter),
and C<sigma> is the width of the neighbourhood as calculated
above (see L<FINDING THE NEIGHBOURS OF THE BMU>). THETA also
decays over time.

The time C<t> is always that of the calling object.

=cut

sub distance_effect { my ($self,$distance,$sigma) = (shift,shift,shift);
	confess "Wrong args" unless defined $distance and defined $sigma;
	return exp (-($distance*$distance) / 2 * ($sigma*$sigma))
}

1;

__END__

=head1 AUTHOR AND COYRIGHT

This implimentation Copyright (C) Lee Goddard, 2003.
All Rights Reserved.

Available under the same terms as Perl itself.
















