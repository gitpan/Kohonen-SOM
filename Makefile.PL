use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    'DISTNAME'	=> 'Kohonen-SOM',
	'ABSTRACT'	=> "Kohonen's Self-organising Feature Maps, slowly...",

    'NAME'		=> 'AI::NeuralNet::Kohonen',
    'VERSION_FROM'	=> 'Kohonen.pm', # finds $VERSION
    'PREREQ_PM'		=> {}, # e.g., Module::Name => 1.1
    ($] >= 5.005 ?    ## Add these new keywords supported since 5.005
      (AUTHOR     => 'Lee Goddard <lgoddard@cpan.org>') : ()),
);
