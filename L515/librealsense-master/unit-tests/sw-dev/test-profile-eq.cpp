// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2026 RealSense, Inc. All Rights Reserved.

#include "../catch.h"
#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>

using namespace rs2;

// video_stream_profile::operator== must additionally compare width and height,
// unlike the base stream_profile::operator== (which only compares index/type/format/fps).
// A software device is used so the test needs no connected hardware.

TEST_CASE( "video_stream_profile operator==", "[software-device]" )
{
    rs2_intrinsics intrinsics{};

    auto dev = std::make_shared< software_device >();
    auto sensor = dev->add_sensor( "software_sensor" );

    // Distinct uids per stream, as a real device would have. uid is not part of either
    // operator== tested below, so it does not affect the comparisons.
    // Two profiles share index/type/format/fps but differ in resolution: equal as base
    // stream_profiles, but must NOT be equal as video_stream_profiles.
    sensor.add_video_stream( { RS2_STREAM_DEPTH, 0, 1, 1280, 720, 30, 2, RS2_FORMAT_Z16, intrinsics } );
    sensor.add_video_stream( { RS2_STREAM_DEPTH, 0, 2,  640, 480, 30, 2, RS2_FORMAT_Z16, intrinsics } );
    // A third profile identical to the first in every attribute except uid (a distinct object).
    sensor.add_video_stream( { RS2_STREAM_DEPTH, 0, 3, 1280, 720, 30, 2, RS2_FORMAT_Z16, intrinsics } );

    std::vector< stream_profile > stream_profiles;
    REQUIRE_NOTHROW( stream_profiles = sensor.get_stream_profiles() );
    REQUIRE( stream_profiles.size() == 3 );

    // Select profiles by their attributes rather than by insertion order.
    video_stream_profile hd, vga, hd_dup;
    for( auto & p : stream_profiles )
    {
        video_stream_profile vp = p.as< video_stream_profile >();
        if( vp.width() == 640 )
            vga = vp;         // the odd-resolution one
        else if( ! hd )
            hd = vp;          // first 1280x720
        else
            hd_dup = vp;      // second 1280x720 (identical attributes, distinct object)
    }
    REQUIRE( hd );
    REQUIRE( vga );
    REQUIRE( hd_dup );

    // Capture results in named bools: video_stream_profile is implicitly convertible to
    // bool, so comparing the profiles inline makes Catch print a misleading "1 == 1".
    // stream_profile::operator== compares only {stream_index, stream_type, format, fps} --
    // not uid or resolution -- so hd and vga are base-equal despite differing in resolution.
    bool base_equal = static_cast< stream_profile & >( hd ) == static_cast< stream_profile & >( vga );
    bool video_equal_diff_resolution = ( hd == vga );
    bool video_equal_same_attributes = ( hd == hd_dup );

    // Same index/type/format/fps -> the base stream_profile comparison treats them as equal
    REQUIRE( base_equal );

    // ... but different width/height -> the video_stream_profile comparison must differ
    REQUIRE_FALSE( video_equal_diff_resolution );

    // Identical attributes including resolution -> video_stream_profile comparison must match
    REQUIRE( video_equal_same_attributes );
}
