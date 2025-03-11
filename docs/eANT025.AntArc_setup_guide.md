# Introduction

This guide is focused on setting up and running our version of a circum-Antarctic NEMO configuration, eANT025.AntArc, on ARCHER2 and assumes that you already have access to the machine. The guide is modified from a version of the "archer2" setup guide and the previous version of the ARCHER2 setup guide (now in: docs/archive/eANT025.L121_archer2_setup_guide.md). Many thanks to Christoph Kittel and Pierre Mathiot who did all the hard work setting up the previous version of this configuration, eANT025.L121, and to Chris Bull for his helpful guides to running WED025 on ARCHER2 (currently offline).

*AntArc = <ins>Ant</ins>hropogenic <ins>AntArc</ins>tic*
# Installing NEMO and other bits

I recommend you install NEMO in your main work directory

    /work/n02/n02/<username>

and save this as an environment variable $WORK (which you'll need later) by adding this line to your `~/.bashrc`:

    export WORK=/work/n02/n02/`whoami`

then do

    source ~/.bashrc
    cd $WORK

Now, download the NEMO model code from the NEMO repository:

    git clone https://forge.nemo-ocean.eu/nemo/nemo.git
    cd nemo

Freeze the code at the 4.2.2 branch (important to freeze it at 4.2.2, since there was a ocean-ice stress bug in 4.2.1!):

    git switch --detach 4.2.2

Copy the Gnu architecture file which was built by the team at NOC (there is also a Cray version but I've had issues with it before):

    cp /work/n01/shared/nemo/ARCH/arch-X86_ARCHER2-Gnu_4.2.fcm arch/

Edit the file `ext/FCM/lib/Fcm/Config.pm`, so that it says:

    FC_MODSEARCH => '-J'

on the relevant line.

Copy another file from the NOC group, which automates the division of processors between NEMO and XIOS:

    cp /work/n01/shared/nemo/mkslurm_hetjob .
    chmod +x mkslurm_hetjob

If you want to change the number of cores you can use this tool to regenerate most of runnemo.sh (a file we'll look at more later). You can learn how to use it [here](https://docs.archer2.ac.uk/research-software/nemo/).

For some of Kaitlin's scripts you will need a python package which is not installed on ARCHER2. Follow the instructions [here](https://docs.archer2.ac.uk/user-guide/python/) to set up a new virtual environment named pyenv, to be stored in $WORK/pyenv. Activate this environment and use pip to install the package f90nml (all explained in the link).

You will need to set up globus-url-copy to transfer data between ARCHER2 and JASMIN, following [these instructions](https://help.jasmin.ac.uk/article/4997-transfers-from-archer2) ("1st choice method"). 

Finally, ARCHER2 has a quirk where it expects booleans, and only booleans, to have trailing commas in the namelists. To fix all the preinstalled namelists (in case you want to run any test configurations later), create edit_nmls and reverse_edit_nmls files following [these instructions](https://forge.ipsl.jussieu.fr/nemo/ticket/2653), and chmod +x them.    

# Adding the new configuration

Copy the configuration setup files (including its custom source code, CPP definitions, and running scripts) from Birgit's shared space to the cfgs/ directory within your NEMO installation:

    cp -r /work/n02/shared/birgal/NEMO_share/AntArc/cfg/ cfgs/AntArc

The input files (atmospheric forcing, boundary conditions, etc) are stored in Birgit's shared space (`/work/n02/shared/birgal/NEMO_share/AntArc/input/`) and these will be linked in when you run a job, so that we don't have to maintain multiple copies.

Also add this configuration to `cfgs/ref_cfgs.txt` so you can use it as a base to compile from, by adding a new line to that file:

    eANT025.AntArc OCE ICE

# Compiling the model

Assuming you're using the Gnu compilers as suggested, you can compile the eANT025.AntArc configuration as follows:

    module swap PrgEnv-cray/8.3.3 PrgEnv-gnu/8.3.3
    module load cray-mpich/8.1.23 
    module load cray-hdf5-parallel/1.12.2.1 
    module load cray-netcdf-hdf5parallel/4.9.0.1
    ./makenemo -m X86_ARCHER2-Gnu_4.2 -r eANT025.AntArc -j 8

(I'm not sure the -j flag does anything as the interactive login nodes are presumably serial!) You will typically only need to recompile when you make changes within the model source code, as most options are set within the namelists.

Once the compilation has completed, save the nemo executable to the EXPREF directory so you don't lose it:

    cp cfgs/eANT025.AntArc/EXP00/nemo cfgs/eANT025.AntArc/EXPREF/

You'll also need to compile an alternate executable with the extra CPP key "key_qco"; we need this just for the first year of the simulation to ensure stability (but it also goes unstable if you use it for too long!) The easiest way to do so is probably:

    ./makenemo -m X86_ARCHER2-Gnu_4.2 -r eANT025.AntArc -n 'eANT025.AntArc_qco' -j 8 add_key 'key_qco'

(You can use this same strategy if you ever want a copy of this configuration with different CPP keys: use the -n flag to set the new name, and add_key or del_key to change the CPP keys.). In this case, move the qco NEMO executable to the original configuration with a new name, and delete the otherwise-unused eANT025.AntArc_qco case:

    mv cfgs/eANT025.AntArc_qco/EXP00/nemo cfgs/eANT025.AntArc/EXPREF/nemo_qco

You will also need to compile the REBUILD_NEMO tool which combines the iceberg processor files into a single iceberg restart file at the end of each run. With the same modules loaded as above, compile the tool:

    cd tools/
    ./maketools -m X86_ARCHER2-Gnu_4.2 -n REBUILD_NEMO

You can follow a similar approach to compile any other tools you might want to use down the line, such as DOMAINcfg or WEIGHTS.

# Running a job

Now let's work within the configuration directory:

    cd cfgs/eANT025.AntArc

The EXPREF/ directory should contain everything you need to run a job. Keep this as a reference directory by never actually running within it and only making copies of it. The compiler will have autocreated an EXP00/ directory, but it will skip some of the important shell scripts in EXPREF/, so best to delete it and make a new one:

    rm -rf EXP00
    cp -r EXPREF NEW_EXP
    cd NEW_EXP

You will run your new experiment within NEW_EXP.

The first step is to link in the forcing files using prepare_run.sh; this also copies the XIOS executable from the NOC group. This script will also set up your namelists for the first year, by calling the python script `update_namelists.py`. If you don't want the first year to be 1979, you'll need to change the first argument to `update_namelists.py` near the bottom of prepare_run.sh. Finally, it will set up a directory on JASMIN where the results will be automatically copied every year. Edit this directory location as needed.

Once you're happy, call

    ./prepare_run.sh

Now, if needed, edit the file `postproc.sh` which will be called after every year of simulation. It will update the namelists to cycle through the years with restarts handled correctly. In this file, check the arguments to `update_namelist.py` to set the start and end years you want for the simulation. The script `postproc.sh` will also copy the results to JASMIN every year (it only transfers those files which don't already exist within the JASMIN directory). If you edited your JASMIN directory before, change it here too so that it matches.

Finally, update the headers in `runnemo_firstyear.sh` and `runnemo.sh` with the budget code that you are charging your CPU hours to:

    #SBATCH --account=project_budget_code

Once you're happy, submit the first year with:

    sbatch runnemo_firstyear.sh

This job script sets off a year-long run of the simulation. Then it will call `postproc.sh`, which will then call `runnemo.sh` and so on until the job is done (i.e. it has reached the end year).

Finally, you should copy the files `prepare_run.sh`, `postproc.sh`, `runnemo.sh` to EXPREF if you made any changes to them that you want to use for all your experiments.

So in summary, all you need to do to run a new experiment NEW_EXP is

    cp -r EXPREF NEW_EXP
    cd NEW_EXP
    <edit prepare_run.sh>
    ./prepare_run.sh
    <edit postproc.sh>
    sbatch runnemo_firstyear.sh

Keep an eye on the ARCHER2 queues with squeue --user=your_username and see what happens in ocean.output. For more detailed information on the queue, I'd recommend adding the following 'sq' alias within your .bashrc file in your home directory: 
    
    alias sq='squeue -o "%.12i %.8u %.9a %.22j %.2t %.10r %.19S %.10M %.10L %.6D %.5C %P %N" -u your_username'

Any problems, contact Birgit (<birgal@bas.ac.uk>) and/or Kaitlin (<kaight@bas.ac.uk>) for advice.
