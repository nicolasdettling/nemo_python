# Introduction

This guide explains how to install and run the circum-Antarctic NEMO configuration eANT025.L121 on ARCHER2. It assumes you already have access to the machine.

Many thanks to Christoph Kittel and Pierre Mathiot who have done all the hard work setting up this configuration, and to Chris Bull for his helpful guides to running WED025 on ARCHER2 (currently offline).

# Installing NEMO and other bits

I recommend you install NEMO in your main work directory

    /work/n02/n02/<username>

and save this as an environment variable $WORK, which you'll need later. Add this line to your ~/.bashrc:

    export WORK=/work/n02/n02/`whoami`

then do

    source ~/.bashrc
    cd $WORK

Now, download the code from the repository:

    git clone https://forge.nemo-ocean.eu/nemo/nemo.git
    cd nemo

Now freeze the code at the 4.2 branch:

    git switch --detach 4.2.0

Copy the Gnu architecture file which was built by the team at NOC (there is also a Cray version but I've had issues with it before):

    cp /work/n01/shared/nemo/ARCH/arch-X86_ARCHER2-Gnu_4.2.fcm arch/

Now edit the file

    ext/FCM/lib/Fcm/Config.pm

so that it says

    FC_MODSEARCH => '-J'

on the relevant line.

Copy another file from the NOC group, which automates the division of processors between NEMO and XIOS:

    cp /work/n01/shared/nemo/mkslurm_hetjob .
    chmod +x mkslurm_hetjob

If you want to change the number of cores you can use this tool to regenerate most of runnemo.sh (a file we'll look at more later). You can learn how to use it [here](https://docs.archer2.ac.uk/research-software/nemo/).

For some of Kaitlin's scripts you will need a python package which is not installed on ARCHER2. Follow the instructions [here](https://docs.archer2.ac.uk/user-guide/python/) to set up a new virtual environment named pyenv, to be stored in $WORK/pyenv. Activate this environment and use pip to install the package f90nml (all explained in the link).

You will need to set up globus-url-copy to transfer data between ARCHER2 and JASMIN, following [these instructions](https://help.jasmin.ac.uk/article/4997-transfers-from-archer2) ("1st choice method"). Make sure you save the credentials file directly within $WORK. Some of Kaitlin's scripts assume your username is the same on ARCHER2 and JASMIN; ask her for help if this is not the case, or just have a go (search for "whoami").

Finally, ARCHER2 has a quirk where it expects booleans, and only booleans, to have trailing commas in the namelists. To fix all the preinstalled namelists (in case you want to run any test configurations later), create edit_nmls and reverse_edit_nmls files following [these instructions](https://forge.ipsl.jussieu.fr/nemo/ticket/2653), and chmod +x them.    

# Adding the new configuration

Copy over the configuration (including its custom source code, CPP defs, and running scripts) from Kaitlin's shared space to your cfgs/ directory:

    cp -r /work/n02/shared/kaight/NEMO_share/cfgs/eANT025.L121 cfgs/

The input files (atmospheric forcing, boundary conditions, etc) are also stored in Kaitlin's shared space (/work/n02/shared/kaight/NEMO_share/input/eANT025.L121 just for information) and these will be linked in when you run a job, so we don't have to maintain multiple copies.

Also add this configuration to cfgs/ref_cfgs.txt so you can use it as a base to compile from. Add a new line to the bottom of that file that says:

    eANT025.L121 OCE ICE

# Compiling the model

Assuming you're using the Gnu compilers as suggested, you can compile the eANT025.L121 configuration as follows:

    module swap PrgEnv-cray/8.3.3 PrgEnv-gnu/8.3.3
    module load cray-mpich/8.1.23 
    module load cray-hdf5-parallel/1.12.2.1 
    module load cray-netcdf-hdf5parallel/4.9.0.1
    ./makenemo -m X86_ARCHER2-Gnu_4.2 -r eANT025.L121 -j 8

(I'm not sure the -j flag does anything as the interactive login nodes are presumably serial!) You might like to save this in a file somewhere so you can automate compiling if needed, although hopefully you won't need to do it too often as most options are set by the namelist.

Now save the executable to the EXPREF directory so you don't lose it:

    cp cfgs/eANT025.L121/EXP00/nemo cfgs/eANT025.L121/EXPREF

Finally, you'll need to compile an alternate executable with the extra CPP key "key_qco"; we need this just for the first year of the simulation to ensure stability (but it also goes unstable if you use it for too long!) The easiest way to do so is probably:

    ./makenemo -m X86_ARCHER2-Gnu_4.2 -r eANT025.L121 -n 'eANT025.L121_qco' -j 8 add_key 'key_qco'

(You can use this same strategy if you ever want a copy of this configuration with different CPP keys: use the -n flag to set the new name, and add_key or del_key to change the CPP keys.)

In this case, move the qco NEMO executable to the original configuration with a new name, and delete the otherwise-unused eANT025.L121_qco case:

    mv cfgs/eANT025.L121_qco/EXP00/nemo cfgs/eANT025.L121/EXPREF/nemo_qco
    rm -rf cfgs/eANT025.L121_qco

# Running a job

Now let's work within the configuration directory:

    cd cfgs/eANT025.L121

The EXPREF/ directory should contain everything you need to run a job. Keep this fresh by never actually running it and only making copies of it. You'll probably have EXP00/ autocreated by the compiler, but it will skip some of the important shell scripts in EXPREF/, so best to delete it and make a new one:

    rm -rf EXP00
    cp -r EXPREF EXP00
    cd EXP00

You will run the experiment within EXP00.

The first step is to link in the forcing files using prepare_run.sh; this also copies the XIOS executable from the NOC group:

    ./prepare_run.sh

This script will also set up your namelists for the first year, by calling the python script update_namelists.py. If you don't want the first year to be 1979, you'll need to change that argument to update_namelists.py near the bottom of prepare_run.sh (the rest of the arguments don't matter here). Finally, it will set up a directory on JASMIN where the results will be automatically copied every year. Edit this directory if needed (for example if you're not in the terrafirma group workspace).

If needed, edit the file postproc.sh which will be called after every year of simulation. It will update the namelists to cycle through the years with restarts handled correctly. In this file, check the arguments to update_namelist.py to set the start and end years you want for the simulation. The script postproc.sh will also copy the results to JASMIN every year and then delete them from ARCHER2 (if all is well) so they don't take up too much space. If you edited your JASMIN directory before, change it here too so it matches.

Once you're happy, submit the first year with:

    sbatch runnemo_firstyear.sh

This job script uses the nemo_qco executable to run one year of simulation. Then it will call postproc.sh, which will then call runnemo.sh (identical to runnemo_firstyear.sh except for the executable with no qco this time) which will call postproc.sh, and so on until the job is done.

So in summary, all you need to do to run a new experiment EXPNEW is

    cp -r EXPREF EXPNEW
    cd EXPNEW
    <edit prepare_run.sh>
    ./prepare_run.sh
    <edit postproc.sh>
    sbatch runnemo_firstyear.sh

and hopefully all should work! Keep an eye on the queues with squeue --user=<username> and see what happens in ocean.output. Any problems, ask Kaitlin for advice on <kaight@bas.ac.uk>.
    