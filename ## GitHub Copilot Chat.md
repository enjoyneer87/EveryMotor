## GitHub Copilot Chat

- Extension: 0.42.3 (prod)
- VS Code: 1.114.0 (e7fb5e96c0730b9deb70b33781f98e2f35975036)
- OS: win32 10.0.19045 x64
- GitHub Account: dhkang87

## Network

User Settings:
```json
  "http.systemCertificatesNode": true,
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.200.245.245 (2 ms)
- DNS ipv6 Lookup: Error (3 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: None (1 ms)
- Electron fetch (configured): HTTP 200 (22 ms)
- Node.js https: HTTP 200 (28 ms)
- Node.js fetch: HTTP 200 (38 ms)

Connecting to https://api.business.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.114.22 (5 ms)
- DNS ipv6 Lookup: Error (38 ms): getaddrinfo ENOTFOUND api.business.githubcopilot.com
- Proxy URL: None (1 ms)
- Electron fetch (configured): HTTP 200 (208 ms)
- Node.js https: HTTP 200 (635 ms)
- Node.js fetch: HTTP 200 (615 ms)

Connecting to https://proxy.business.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 52.175.140.176 (4 ms)
- DNS ipv6 Lookup: Error (5 ms): getaddrinfo ENOTFOUND proxy.business.githubcopilot.com
- Proxy URL: None (54 ms)
- Electron fetch (configured): HTTP 200 (21 ms)
- Node.js https: HTTP 200 (87 ms)
- Node.js fetch: HTTP 200 (82 ms)

Connecting to https://mobile.events.data.microsoft.com: HTTP 404 (132 ms)
Connecting to https://dc.services.visualstudio.com: HTTP 404 (637 ms)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: HTTP 200 (631 ms)
Connecting to https://telemetry.business.githubcopilot.com/_ping: HTTP 200 (609 ms)
Connecting to https://default.exp-tas.com: HTTP 400 (116 ms)

Number of system certificates: 71

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).