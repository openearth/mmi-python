import logging

import numpy as np
import zmq
import osgeo.osr
import shapely.geometry

from . import send_array, recv_array

logging.basicConfig()
logger = logging.getLogger(__name__)


class Views(object):
    # TODO: rewrite using config file per engine
    @staticmethod
    def grid(context):
        meta = context["value"]
        # Get connection info
        node = meta['node']
        node = 'localhost'
        req_port = meta["ports"]["REQ"]
        ctx = context["ctx"]
        req = ctx.socket(zmq.REQ)
        req.connect("tcp://%s:%s" % (node, req_port))
        # Get grid variables
        send_array(req, metadata={"get_var": "xk"})
        xk, A = recv_array(req)
        send_array(req, metadata={"get_var": "yk"})
        yk, A = recv_array(req)
        # Spatial transform
        points = np.c_[xk, yk]
        logger.info("points shape: %s, values: %s", points.shape, points)
        src_srs = osgeo.osr.SpatialReference()
        src_srs.ImportFromEPSG(meta["epsg"])
        dst_srs = osgeo.osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        transform = osgeo.osr.CoordinateTransformation(src_srs, dst_srs)
        wkt_points = transform.TransformPoints(points)
        geom = shapely.geometry.MultiPoint(wkt_points)

        geojson = shapely.geometry.mapping(geom)
        return geojson


views = Views()
