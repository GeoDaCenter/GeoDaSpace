""" Various functions to facilitate the use of the GeoDa Center Xgrid.

Typical work flow might be as follows:

import grid_loader as GL
my_jobs = ['la', 'nyc', 'chi', 'sf']
job_ids = []
for job in my_jobs:
    xgrid_id = GL.load_python('compute_stuff.py', job, pysal=True)
    job_ids.append([xgrid_id, job])
GL.write_jids(job_ids, 'saved_ids.csv')
GL.results_runner(job_ids, 'results_folder/')

The script compute_stuff.py is the one sent out to the agents. This might
contain the simulation code. In the example, compute_stuff.py takes a
command line argument.

"""

__author__ = "David Folch david.folch@asu.edu, Pedro Amaral pedro.amaral@asu.edu"

import subprocess as SP
import time


def load_general(command):
    """ 
    Load a single job onto the GeoDa Center Xgrid. Note: there is a 
    utility function called load_python() that makes it easier to pass a
    python specific job.

    Parameters
    ----------

    command     : string
                  Identical to what would passed at the command line to
                  execute an Xgrid job

    Returns
    -------

    jid         : string
                  Job ID created by the Xgrid controller for tracking the job

    """
    stdout = _send(command)
    jid = stdout.replace('=', '|')
    jid = jid.replace(';', '|')
    jid = jid.split('|')
    jid = jid[1].replace(' ', '')
    return jid

def load_python(py_script, args="", other_folders="", pysal='False'):
    """ 
    Load a single python job onto the GeoDa Center Xgrid. Forces the
    job to run on 32 bit Enthought Python on each agent machine.

    Parameters
    ----------

    py_script   : string
                  File name of python script to be executed
    args        : string
                  Optional arguments to pass to the python script
    folders     : string
                  Folder names containing content needed to execute the Xgrid
                  job. These might include data folders or python libraries
                  needed for the job.
    pysal       : boolean
                  If true append PySAL to the job. This assumes the pysal
                  file tree is located in the working directory. Note: the
                  'examples' folder should be deleted from pysal before 
                  starting the job.

    Returns
    -------

    jid         : string
                  Job ID created by the Xgrid controller for tracking the job
    
    """

    command = 'xgrid -h tethys.asu.edu -p coorhall -job submit /Library/Frameworks/Python.framework/Versions/Current/bin/python %s %s %s ' % (py_script, args, other_folders)
    if pysal:
        command += 'pysal/'
    return load_general(command)


def results(jids, outfolder):
    """ Checks the status of each remaining job. When a job is complete it 
    pulls the results to the client machine and deletes the job from the 
    controller. It also prints (interesting) information about the completed
    jobs and the remaining jobs.

    Parameters
    ----------

    jids        : list of lists
                  Each element of list must contain a list (or tuple), where
                  the first element is an Xgrid job ID as a string; all other
                  elements can be user defined identifiers for that job. For
                  example [('345', 'chicago'), ('346', 'phoenix'), ('347',
                  'denver')].
    outfolder   : string
                  Folder where job results should be placed

    """
    
    done = []
    for i in jids:
        command = 'xgrid -h tethys.asu.edu -p coorhall -job log -id %s' % i[0]
        stdout = _send(command)
        if "Finished" in stdout:
            command = 'xgrid -h tethys.asu.edu -p coorhall -job results -id %s -out %s' % (i[0], outfolder)
            _send(command,get=0)
            print 'finished', i
            time.sleep(1)
            _cleanup_results(i)
            done.append(i)
        elif "Failed" in stdout:
            print '**** failed', i
            time.sleep(1)
            _cleanup_results(i)
            done.append(i)
    for i in done:        
        jids.remove(i)
    if len(jids) == 0:
        pass
    elif len(jids) < 10:
        print 'remaining jobs'
        for i in jids:
            agent = _get_details(i)
            print '\t', i, agent
        print '\n\n'
    else:
        print 'remaining jobs', len(jids), '\n\n'
    return jids


def results_runner(jids, outfolder, delay=600):
    """ Continues to check the status of each remaining job until all jobs are
    complete. When a job is complete it pulls the results to the client machine 
    and deletes the job from the controller.

    Parameters
    ----------

    jids        : list of lists
                  Each element of list must contain a list (or tuple), where
                  the first element is an Xgrid job ID as a string; all other
                  elements can be user defined identifiers for that job. For
                  example [('345', 'chicago'), ('346', 'phoenix'), ('347',
                  'denver')].
    outfolder   : string
                  Folder where job results should be placed
    delay       : integer
                  Number of seconds to wait before checking the status of the
                  remaining jobs

    """
    while jids:
        time.sleep(delay)
        jids = results(jids, outfolder)
    

def write_jids(jids, outfile):
    """ Write out Xgrid job IDs and user defined identifiers of the job to the
    hard drive. Some jobs take awhile and this provides a record in case
    some unexpected event causes the loss of Xgrid job IDs that might only be
    stored in RAM.

    Parameters
    ----------

    jids        : list of lists
                  Each element of list must contain a list (or tuple), where
                  the first element is an Xgrid job ID as a string; all other
                  elements can be user defined identifiers for that job. For
                  example [('345', 'chicago'), ('346', 'phoenix'), ('347',
                  'denver')].
    outfile     : string
                  File name for storing job information.
    
    """
    outfile = open(outfile, 'w')
    for line in jids:
        temp = ''
        for i in line:
            temp += str(i) + ','
        outfile.write(temp[:-1] +'\n')
    outfile.close()



##############################################
###### Internal Functions Below Here #########
##############################################

def _cleanup_results(jid):
    agent = _get_details(jid)
    print '\tagent name:', agent
    #print '\ttotal time (minutes):', tot_time/60.0
    command = 'xgrid -h tethys.asu.edu -p coorhall -job delete -id %s' % jid[0]
    _send(command, get=0)

def _get_details(jid):
    command = 'xgrid -h tethys.asu.edu -p coorhall -job log -id %s' % jid[0]
    stdout = _send(command)
    return _parse_log(stdout)

def _parse_log(log):
    while 'message = ' in log:
        log = log.replace('message = ', '|M_start')
    while 'time = ' in log:
        log = log.replace('time = ', '|T_start')
    log = log.replace(';', '|')
    log = log.replace('"', '')
    log = log.replace('\\', '')
    log = log.split('|')
    messages = []
    #time2 = None
    #tot_time = None
    agent = 'No agent assigned'
    #times = []
    for i in log:
        if 'M_start' in i:
            messages.append(i[7:])
        #elif 'T_start' in i:
            #times.append(i[7:])
    for i in range(len(messages)):
        if 'submitted to agent ' in messages[i]:
            agent = messages[i][26:]
            #time1 = int(times[i])
        #elif 'job state changed to Finished' in messages[i]:
            #time2 = int(times[i])
        #elif 'job state changed to Failed' in messages[i]:
            #time2 = int(times[i])
    #start_time = time.ctime(time1 + 978307200)
    #if time2:
        #tot_time = time2 - time1
    return agent

def _send(command,get=1):
    process = SP.Popen(command, shell=True, stdout=SP.PIPE)
    if get==1:
        stdout, stderr = process.communicate()
        return stdout

