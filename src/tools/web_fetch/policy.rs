use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use reqwest::Url;

/// Validate a URL and resolve it to public addresses. The returned addresses
/// are pinned into the request client so DNS cannot change between validation
/// and connection.
pub async fn validate_destination(url: &Url) -> Result<Vec<SocketAddr>, String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("URL must use http:// or https://".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URLs containing credentials are not allowed".to_string());
    }

    let host = url
        .host_str()
        .ok_or_else(|| "URL must include a host".to_string())?;
    let normalized_host = host.trim_end_matches('.').to_ascii_lowercase();
    if normalized_host == "localhost"
        || normalized_host.ends_with(".localhost")
        || normalized_host.ends_with(".local")
        || normalized_host.ends_with(".internal")
    {
        return Err(format!("Private destination is not allowed: {host}"));
    }

    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL has no usable port".to_string())?;
    let addresses: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("Failed to resolve {host}: {e}"))?
        .collect();

    if addresses.is_empty() {
        return Err(format!("Host did not resolve: {host}"));
    }
    if let Some(address) = addresses.iter().find(|addr| !is_public_ip(addr.ip())) {
        return Err(format!(
            "Private or non-public destination is not allowed: {}",
            address.ip()
        ));
    }

    Ok(addresses)
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => is_public_ipv6(ip),
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, d] = ip.octets();
    !matches!(
        (a, b, c, d),
        (0, _, _, _)
            | (10, _, _, _)
            | (100, 64..=127, _, _)
            | (127, _, _, _)
            | (169, 254, _, _)
            | (172, 16..=31, _, _)
            | (192, 0, 0, _)
            | (192, 0, 2, _)
            | (192, 168, _, _)
            | (198, 18..=19, _, _)
            | (198, 51, 100, _)
            | (203, 0, 113, _)
            | (224..=255, _, _, _)
    )
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    if ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        // fc00::/7 unique local
        || (segments[0] & 0xfe00) == 0xfc00
        // fe80::/10 link-local
        || (segments[0] & 0xffc0) == 0xfe80
        // fec0::/10 site-local (deprecated but still routed by some stacks)
        || (segments[0] & 0xffc0) == 0xfec0
        // 2001:db8::/32 documentation
        || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        // 2001::/32 Teredo: tunnels to an arbitrary IPv4 host
        || (segments[0] == 0x2001 && segments[1] == 0x0000)
        // 2001:2::/48 benchmarking
        || (segments[0] == 0x2001 && segments[1] == 0x0002 && segments[2] == 0x0000)
        // 2001:10::/28 and 2001:20::/28 ORCHID
        || (segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0010)
        || (segments[0] == 0x2001 && (segments[1] & 0xfff0) == 0x0020)
        // 64:ff9b:1::/48 local-use NAT64
        || (segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2] == 0x0001)
    {
        return false;
    }

    // Transition prefixes embed an IPv4 address; judge the embedded one so
    // a private IPv4 target cannot be reached through the IPv6 form.
    if let Some(ipv4) = embedded_ipv4(&segments) {
        return is_public_ipv4(ipv4);
    }
    true
}

/// The IPv4 address carried by an IPv4-mapped, IPv4-compatible, NAT64, or
/// 6to4 IPv6 address.
fn embedded_ipv4(segments: &[u16; 8]) -> Option<Ipv4Addr> {
    let from_pair = |high: u16, low: u16| {
        Ipv4Addr::new(
            (high >> 8) as u8,
            (high & 0xff) as u8,
            (low >> 8) as u8,
            (low & 0xff) as u8,
        )
    };
    match segments {
        // ::ffff:a.b.c.d (IPv4-mapped)
        [0, 0, 0, 0, 0, 0xffff, high, low] => Some(from_pair(*high, *low)),
        // ::a.b.c.d (IPv4-compatible, deprecated); :: and ::1 are handled
        // earlier by is_unspecified/is_loopback
        [0, 0, 0, 0, 0, 0, high, low] => Some(from_pair(*high, *low)),
        // 64:ff9b::a.b.c.d (well-known NAT64 prefix)
        [0x0064, 0xff9b, 0, 0, 0, 0, high, low] => Some(from_pair(*high, *low)),
        // 2002:abcd:efgh::/48 (6to4)
        [0x2002, high, low, ..] => Some(from_pair(*high, *low)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_public_ipv4_ranges() {
        for ip in [
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "172.16.0.1",
            "192.168.1.1",
            "224.0.0.1",
        ] {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip}");
        }
        assert!(is_public_ip("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn rejects_non_public_ipv6_ranges() {
        for ip in [
            "::",
            "::1",
            "fc00::1",
            "fe80::1",
            "fec0::1",
            "2001:db8::1",
            "2001::1",
            "2001:2::1",
            "2001:10::1",
            "2001:2f::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            // IPv4-compatible
            "::10.0.0.1",
            "::a00:1",
            // NAT64 well-known and local-use prefixes
            "64:ff9b::10.0.0.1",
            "64:ff9b::a00:1",
            "64:ff9b:1::8.8.8.8",
            // 6to4 carrying private IPv4
            "2002:a00:1::1",
            "2002:c0a8:101::1",
        ] {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "2606:4700:4700::1111",
            "::ffff:8.8.8.8",
            "64:ff9b::8.8.8.8",
            "2002:808:808::1",
            "2001:4860:4860::8888",
        ] {
            assert!(is_public_ip(ip.parse().unwrap()), "{ip}");
        }
    }

    #[tokio::test]
    async fn rejects_loopback_literal_before_connecting() {
        let url = Url::parse("http://127.0.0.1/admin").unwrap();
        let error = validate_destination(&url).await.unwrap_err();
        assert!(error.contains("non-public"));
    }

    #[tokio::test]
    async fn rejects_localhost_name_before_resolving() {
        let url = Url::parse("http://localhost/admin").unwrap();
        let error = validate_destination(&url).await.unwrap_err();
        assert!(error.contains("Private destination"));
    }
}
