#-------------------------------------
# change data into matlab sparse matrix format
#-------------------------------------

use strict;
use warnings;

die "Usage:  perl  matlab_format.pl  input_file sif_file\n" if (@ARGV != 2);

my $input_file = $ARGV[0];
my $sif_file=$ARGV[1];

my %sif;
open(I, $sif_file) || die "Cannot open $sif_file\n";
  while (my $str = <I>) {
    $str =~ s/\s+$//;
    my @tokens = split(/\s+/, $str);
    $sif{$tokens[0]}=$tokens[1];
}
close(I);


open(I, $input_file) || die "Cannot open $input_file\n";
open(O, ">$input_file.ml") || die "Cannot create $input_file.ml\n";
my $line = 1;
while (my $str = <I>) {
    $str =~ s/\s+$//;
    my @tokens = split(/\s+/, $str);
    my $l = @tokens;
    for (my $i = 0; $i < $l; $i++) {
          my $sif_v=1;
          if (exists $sif{$tokens[$i]}){ $sif_v=$sif{$tokens[$i]};}
	  print O "$tokens[$i]\t$line\t$sif_v\n";
    }
   $line++; 
}
close(I);
close(O);
