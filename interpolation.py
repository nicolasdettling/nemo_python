from .utils import polar_stereo_inv

# Helper function to construct a minimal CF field so cf-python can do regridding.
# Mostly following Robin's Unicicles coupling code in UKESM.
# If x and y are polar stereographic rather than lon-lat, pass polar_stereo=True.
def construct_cf (data, x, y, polar_stereo=False):

    import cf
    field = cf.Field()


def interp_latlon_cf ():
