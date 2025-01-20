#!/usr/bin/perl
use Getopt::Std;

###############################################################################
#
# catsearch.pl
#
# Author: Chenggang Yu
# Biotechnology HPC Software Applications Institute, 
# Telemedicine and Advanced Technology Research Center, 
# U.S. Army Medical Research and Materiel Command, Fort Detrick, MD, USA
#
# Date: April, 2008
#
# The program predicts query proteins' catalytic functions by searching the 
# CatFam enzyme profile database. The program reads the query proteins from a 
# fasta format file and predict Enzyme Commission (EC) numbers for each query
# that is predicted as enzymes. The profile database search is powered by 
# NCBI's rpsblast, which has to be installed properly. Please read the README
# file for more information.
#
# Disclaimer
# The opinions and assertions contained herein are the private views of the 
# authors and are not to be construed as official or as reflecting the views 
# of the U.S. Army or of the U.S. Department of Defense.
###############################################################################


my %opts=();
getopts('i:d:o:vh',\%opts);

sub usage {
	print "\ncatsearch.pl arguments:\n\n";	
	print "i	input fasta file\n";
	print "d	the CatFam database (default is ./CatFamDB/CatFam4D99R)\n";
	print "o    	the output file (standard output if not specified)\n\n";
}


if($opts{h}) {
	usage();
	exit;
}
							
my $db="./CatFamDB/CatFam4D99R";  #default CatFam database
$db=$opts{d}	if(exists $opts{d}); 

my @dbfiles=glob("$db.*");
my $numf=scalar(@dbfiles);
if($numf<1) {
	print "!!!Cannot find the specified (or default) profile database $db.\n"; 
	usage();
	exit;
}

my $fn=$opts{i};
unless(-f $fn) {
	print "!!!Not able to read input fasta file $fn. Please specify the correct file name.\n";
	usage();
	exit;
}
open(FP, "-|", "rpsblast -d $db -i $fn") or die "!!!Cannot run program rpsblast. You have to install rpsblast first and allow catsearch to call the program.\n";


if(exists $opts{o}) {
	open($FO, ">$opts{o}") or die "Cannot open the output file $opts{o}\n";
} else {
	$FO=STDOUT;	
}

my @ftext=<FP>;

my @results=getRPSHit("RScore", "mb", @ftext);
my $query;
my @out=();
my %myec=();
my @querys=();
foreach my $ln (@results) {
	if($ln=~m/^>QUERY\s*(\S+)\b/) { $query=$1; %myec=();  push @querys, $1; }
	
	if($ln=~m/^>>(.*)/) {
		my @lst=split /\s+/, $1;
		if($lst[2] eq '+') {
			push @out, [$query, $lst[1]] unless(exists $myec{$lst[1]});
			$myec{$lst[1]}=1;
		}
	}
}
my $i=0;
foreach my $query (@querys) {
	if($query ne $out[$i][0]) {
		print $FO "$query\tN/A\n";
	} else {
		while($out[$i][0] eq $query) {
			print $FO "$out[$i][0]\t$out[$i][1]\n";
			$i++;
		}
	}
}
close(FO);

#########################################################################################
sub getRPSHit()
{
my $alist;
my @outlist=();
my %ulist=(); #save unique hit name
my $hitName; 
my $iter=0;
my $lstart=0;
my $bline;
my $fieldName=@_[0];
my $blastFormat=@_[1];
my @ftext=@_[2..$#_];
my $flen=scalar(@ftext);
for(my $i=0; $i<$flen; $i++) {
	if($ftext[$i]=~m/^Query=\s*(\S+)\s+/) {
		%ulist=();
		push @outlist, ">QUERY ".$1;
		for(my $j=$i+1; $j<$flen; $j++) {
		
			if($ftext[$j]=~m/^Query=/) {
				$i=$j-1;
				last;
			}
			my ($id, $fec, $tf, $ps, $st, $sc);
			if($ftext[$j]=~m/^>.*EPD\|(\d+)\b/) { #get one hit
				$id=$1;
				if(not exists $ulist{$id}) {
					$ulist{$id}=1; # a EPD id is saved in the hash table
					my $str=$ftext[$j];
					my $m=1;
					while($str!~m/PS=/) {
						chomp $str;
						$str=$str." ".$ftext[$j+$m];
						$m++;
					}
					chomp $str;
					$str=$str." ".$ftext[$j+$m];
					if($str=~m/SCORE_TH(\d+)\b.*FEC(\d+(\.\d+){1,3}),.*PS=(\w+_\w+)\b/) {
						$st=$1;
						$fec=$2;
						$ps=$4;
						#print "ID=$id FEC=$fec PS=$ps STH=$st\n";
					} else {
						print "Not all info is provided for hit $sc\n";
					}
					my $hitname=$1;
					for(my $k=$j+1; $k<$flen; $k++) {
						if($ftext[$k]=~m/^>/) {
							$j=$k-1;
							last;
						}
						$sc=-1;
						if($fieldName eq "Evalue") { 
							if($ftext[$k]=~m/Expect\s*=\s*(\S+)/) {
								$sc=$1;
							}
						}
						if($fieldName eq "Score") {
							if($ftext[$k]=~m/Score\s*=\s*(\d+)/) {
								$sc=$1;
							}
						}
						if($fieldName eq "RScore") {
							if($ftext[$k]=~m/Score\s*=[^(]*\((\d+)\)/) {
								$sc=$1;
							}
						}
						if($sc>=$st) {
							push @outlist, ">>$id $fec + $ps $st $sc";
							last;
						} elsif($sc>-1) {
							push @outlist, ">>$id $fec - $ps $st $sc";
							last;
						}
					}
				}
			}	
		}
	} else { next; }
}
return @outlist;
}

