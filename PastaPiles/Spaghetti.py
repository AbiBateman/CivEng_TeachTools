"""
Authors: A.H. Bateman and D.J. White

Assoicated publication:
Bateman, A. H and White D. J. (2026)
The brittle response of axially-loaded piles: Pysical models for teaching.
Proceedings of the 11th International Conference on Physical Modelling in Geotechnics. Zurich, Switzerland.

This python script analyses a theoretical pasta pile model given known spaghetti and model parameters.

This code is available at the following repository: https://github.com/AbiBateman/CivEng_TeachTools

This code is tested on Python version 3.12.3.

*** NOTES ***
This python code 

currently only one spaghetti object can be defined per model

*** DISCLAIMER ***
This code is provided by the authors to show the theoretical analysis of the pasta pile model.
It is not intended to be used or relied upon (in whole or part) for any other purpose, and no warranty is provided or implied.
While every effort has been made, the authors cannot guarentee that this code is error free.

*** LICENSE ***
Copyright (c) 2026 Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = "1.0.0"

# builtin imports
import numpy as np  ## tested version 1.26.4
# if script is run directly, matplotlib (tested version 3.9.2) will be imported later

def example():
    """
    Function defines in input parameters and runs the analysis for an example problem.
    A plot of force vs displacement is produced.
    """

    figname = "PastaPiles_LoadDisplacement.svg"  ## where to save the output figure

    ## RUN ANLYSIS WITH INPUT PARAMETERS

    # Create spaghetti object with properties attached
    spag_ex = Spaghetti(
        D = 1.66,  ## Diameter of the spaghetti [mm]
    ) ## Create a spaghetti object
    spag_ex.three_point_bending(
        L = 100,  ## Distance between supports
        K = 0.2,  ## Stiffness of result from three-point bending (F/w) [N/mm]
        MaxForce = 0.9,  ## Maximum force reached in the three-point bending test [N]
    ) ## Input of three-point bending results to estimate sig_f and E

    # Create the model analysis
    model = Spag_Pile(
        spaghetti = spag_ex,  ## defines spaghetti object
        no_layers = 9,  ## Number of layers of spaghetti
        support_dist = 100,  ## Distance between spaghetti supports [mm]
        k_p = 11,  ## Stiffness of a singular pile spring (Eq. 10) [N/mm]
        rigid = False,  ## Pile is rigid (True) or flexible (False)
    )

    # Extract data
    print("Maximum load capacity of a rigid pile (N)")
    print(model.rigid_max_load)
    print("Maximum displacement of a rigid pile (mm)")
    print(model.Swf)
    print("Maximum load reached on a flexible pile (N)")
    print(model.flex_force[1])

    # Plot Results
    import matplotlib.pyplot as plt
    plt.plot(model.flex_disp, model.flex_force)
    plt.axhline(model.rigid_max_load)
    plt.ylim(0); plt.xlim(0)
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.savefig(figname)

class Spaghetti():
    """Defines the properties of the spaghetti"""

    def __init__(
        self,
        D: float,  ## Diameter of spaghetti [mm]
        sig_f: float = 50,  ## Flexural stress at failure [N/mm^2] with predefined value
        E: float = 15000,  ## Young's Modulus [N/mm^2] with predefined value
    ):
        """Defines the diameter and second moment of area of the spaghetti"""
        self.D = D
        self.I = (np.pi / 4) * (self.D / 2) ** 4  ## Second moment of area of a spaghetti [m^4]
        self.sig_f = sig_f
        self.E = E

    def three_point_bending(
            self,
            L: float,  ## Length of specimen between supports [mm]
            K: float,  ## Stiffness of the test [N/mm]
            MaxForce: float,  ## Maximum force reached [N]
    ):
        """Calculates the E and sig_f based on a three point bending test"""
        self.E = K * L**3 / (48 * self.I)  ## Young's modulus [N/mm2] (overwrites existing value) (Eq. 13)
        self.sig_f = MaxForce * L / (np.pi * (self.D/2)**3)  ## Flexurate stress at failure (overwrites existing value) (Eq. 16)

class Spag_Pile():
    """Calculates the load-displacement response for a pasta pile"""

    def __init__(
            self,
            spaghetti: Spaghetti,  ## Object Spaghetti defining spaghetti (t-z) parameters
            no_layers: int,  ## Number of layers of spaghetti (including base spaghetti)
            support_dist: float,  ## Distance between spaghetti supports
            k_p: float,  ## Stiffness of one segment of a pile [N/mm]
            rigid: bool,  ## Pile is rigid (True) or flexible (False)
    ):
        self.spaghetti = spaghetti
        self.no_layers = no_layers
        self.N = self.no_layers - 1  ## N is the number of pile springs in the model
        self.support_dist = support_dist
        self.k_spag = 48 * self.spaghetti.E * self.spaghetti.I / self.support_dist**3
        self.k_p = k_p
        self.rigid = rigid

        # Calculate the failure load and displacement of a single spaghetti in the model
        self.SPf, self.Swf = self.failure()

        # Calculate the rigid pile failure load value
        self.rigid_max_load = self.no_layers * self.SPf  ## Rigid pile failure [N]

        if not rigid:
            # Calculate the stiffness variation with depth of a flexible pile
            self.stiffness_depth, self.stiffness = self.flex_stiffness_depth()  ## stiffness at different depths and at the head
            # Calculate the force vs displacement plot
            self.flex_disp, self.flex_force, self.multiple_failures = self.flex_failure()
            # Calculate Rpf
            self.Rpf = max(np.array(self.flex_force) / self.rigid_max_load)

        self.C = self.C_func()


    def failure(
            self,
    ):
        """Calculate the failure load and displacement of an individual spaghetti under three point bending"""
        SPf = self.spaghetti.sig_f * np.pi * (self.spaghetti.D / 2) ** 3 / self.support_dist ## failure load [N] (Eq. 16)
        Swf = SPf * self.support_dist**3 / (48 * self.spaghetti.E * self.spaghetti.I) ## failure displacement [mm] (Eq. 13)

        return SPf, Swf

    def flex_stiffness_depth(
            self,
    ):
        """Calculate the stiffness, displacement and force with depth"""
        ## Calculation completed bottom up
        k_depth = {}
        k_head = []

        for no_failures in range(self.no_layers):

            ## Select stiffness of lowest spaghetti
            current = self.k_spag

            ## Define k_depth as a list of stiffness along the length of the pile for each failure
            k_depth[no_failures] = [current]

            for current_layer in range(self.no_layers-1):
                current_layer = self.no_layers - 2 - current_layer

                ## bottom spaghetti is in series with the pile above
                current = 1/(1/current + 1/self.k_p)
                k_depth[no_failures].append(current)

                ## If next spaghetti layer has failed, stiffness remains unchanged
                if current_layer < no_failures:
                    k_depth[no_failures].append(current)
                    continue

                ## If no failure, spaghetti is in parallel with pile below
                current = self.k_spag + current
                k_depth[no_failures].append(current)

            k_head.append(current)

            ## Reverse order so head reponse is [0]
            k_depth[no_failures].reverse()

        return k_depth, k_head
    
    def flex_failure(
            self,
    ):
        """Calulate the force and displacement for each spaghetti failure"""

        ## Initialise with first failure locations
        disp = [0, self.Swf, self.Swf]
        force = [0, self.Swf * self.stiffness_depth[0][0], self.Swf * self.stiffness_depth[1][0]]

        disp_save = disp[-1]
        mutliple_failures = 1

        for no_failures in range(self.no_layers-1):
            no_failures += 1

            ## Just before failure
            current_disp = self.Swf  ## displacement of spaghetti about to fail
            current_force = current_disp * self.stiffness_depth[no_failures][no_failures*2]  ## force in spaghetti about to fail (same as force at head)
            disp_head = current_force / self.stiffness_depth[no_failures][0]  ## displacement at pile head
            if disp_head < disp_save:
                disp.append(np.NaN)
                mutliple_failures += 1
                # print(f"{no_failures+1}th spaghetti failure happens simultaneously with previous failure")
            else:
                disp.append(disp_head)
                disp_save = disp_head
            force.append(current_force)

            if no_failures == self.no_layers-1:
                break

            ## Just after failure
            disp.append(disp_head)
            force.append(disp_head * self.stiffness_depth[no_failures+1][0])

        return disp, force, mutliple_failures
    
    def C_func(
            self,
    ):
        """Calculate the pile compliance"""
        return self.SPf * self.no_layers / (self.k_p / (self.no_layers-1)) / self.Swf
    
if __name__ == "__main__":
    # script was run
    example()