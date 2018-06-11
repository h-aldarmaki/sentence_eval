#---------------------------------------------------------
# lowercase all characters and noralize digits
#---------------------------------------------------------

use strict;
use warnings;

die "perl clean.pl input_file output_file\n" if (@ARGV != 2);

my $input_file = $ARGV[0];
my $output_file = $ARGV[1];

open(I, $input_file) || die "Cannot open $input_file\n";
open(O, ">$output_file") || die "Cannot create $output_file\n";

my $i=1;
while (my $str = <I>) {
  $str =~ s/\s+$//;
  if ($str eq "") { next;}
  foreach my $token (split(/\s+/, $str)) {
    $token =~ s/\d+/<num>/g;
    $token = lc $token;
    print O "$token ";
  }
  print O "\n";
  $i++;
}

close(I);
close(O);
