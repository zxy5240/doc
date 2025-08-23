# 下载

### 最新发布

- [Carla 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) - [文档](https://carla.readthedocs.io/en/0.9.15/)

### 每晚构建

这是一个自动构建，最新的更改已推送到我们的 `ue4-dev` 分支。它包含将成为下一个版本的一部分的最新修复和功能，但也包含一些实验性更改。使用风险自负！

- [Carla 每晚构建 (Linux)](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/Dev/CARLA_Latest.tar.gz) 
- [附加地图夜间构建 (Linux)](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/Dev/AdditionalMaps_Latest.tar.gz)
- [Carla 每晚构建（Windows）](https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/Dev/CARLA_Latest.zip) 
- [附加地图每晚构建 (Windows)](https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/Dev/AdditionalMaps_Latest.zip)

<p><a id="last-run-link" href='https://github.com/carla-simulator/carla/actions'>Last successful build</a>: <span id="last-run-time" class="loading">Loading...</span></p>

### 版本 0.9.x

> 以下是 Carla 的先前版本，其中包含每个版本的特定文档的链接：

- [Carla 0.9.14](https://github.com/carla-simulator/carla/releases/tag/0.9.14/) - [文档](https://carla.readthedocs.io/en/0.9.14/)
- [Carla 0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13/) - [文档](https://carla.readthedocs.io/en/0.9.13/)
- [Carla 0.9.12](https://github.com/carla-simulator/carla/releases/tag/0.9.12/) - [文档](https://carla.readthedocs.io/en/0.9.12/)
- [Carla 0.9.11](https://github.com/carla-simulator/carla/releases/tag/0.9.11/) - [文档](https://carla.readthedocs.io/en/0.9.11/)
- [Carla 0.9.10](https://github.com/carla-simulator/carla/releases/tag/0.9.10/) - [文档](https://carla.readthedocs.io/en/0.9.10/)
- [Carla 0.9.9](https://github.com/carla-simulator/carla/releases/tag/0.9.9/) - [文档](https://carla.readthedocs.io/en/0.9.9/)
- [Carla 0.9.8](https://github.com/carla-simulator/carla/releases/tag/0.9.8/) - [文档](https://carla.readthedocs.io/en/0.9.8/)
- [Carla 0.9.7](https://github.com/carla-simulator/carla/releases/tag/0.9.7/) - [文档](https://carla.readthedocs.io/en/0.9.7/)
- [Carla 0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6/) - [文档](https://carla.readthedocs.io/en/0.9.6/)
- [Carla 0.9.5](https://github.com/carla-simulator/carla/releases/tag/0.9.5/) - [文档](https://carla.readthedocs.io/en/0.9.5/)
- [Carla 0.9.4](https://github.com/carla-simulator/carla/releases/tag/0.9.4/) - [文档](https://carla.readthedocs.io/en/0.9.4/)
- [Carla 0.9.3](https://github.com/carla-simulator/carla/releases/tag/0.9.3/) - [文档](https://carla.readthedocs.io/en/0.9.3/)
- [Carla 0.9.2](https://github.com/carla-simulator/carla/releases/tag/0.9.2/) - [文档](https://carla.readthedocs.io/en/0.9.2/)
- [Carla 0.9.1](https://github.com/carla-simulator/carla/releases/tag/0.9.1/) - [文档](https://carla.readthedocs.io/en/0.9.1/)
- [Carla 0.9.0](https://github.com/carla-simulator/carla/releases/tag/0.9.0/) - [文档](https://carla.readthedocs.io/en/0.9.0/)

### Versions 0.8.x

- [Carla 0.8.4](https://github.com/carla-simulator/carla/releases/tag/0.8.4/) - [文档](https://carla.readthedocs.io/en/0.8.4/)
- [Carla 0.8.3](https://github.com/carla-simulator/carla/releases/tag/0.8.3/)
- [Carla 0.8.2](https://github.com/carla-simulator/carla/releases/tag/0.8.2/) - [文档](https://carla.readthedocs.io/en/stable/)

- - -

### Docker

所有版本都可以从 DockerHub 中获取：

```sh
docker pull carlasim/carla:X.X.X
```

使用标签“latest”表示最新版本：

```sh
docker pull carlasim/carla:latest
```


<script>
async function getLastWorkflowRun(owner, repo, workflowFileName) {
  const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflowFileName}/runs?status=completed&per_page=1`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/vnd.github.v3+json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    if (data.workflow_runs && data.workflow_runs.length > 0) {
      const lastRun = data.workflow_runs[0];
      return {
        timestamp: lastRun.updated_at,
        url: lastRun.html_url,
        status: lastRun.conclusion
      };
    }
    return null;
  } catch (error) {
    console.error('Error fetching workflow runs:', error);
    return null;
  }
}

// Format timestamp to be more readable
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
    });
}

// Example usage
getLastWorkflowRun('carla-simulator', 'carla', 'ue4_dev.yml')
  .then(result => {
    if (result) {
      console.log('Last successful run:', result.timestamp);
      console.log('View run:', result.url);
      const lastRunTimeElement = document.getElementById('last-run-time');
      const lastRunLink = document.getElementById('last-run-link')
      //const lastRun = result.workflow_runs[0];
      const formattedTime = formatTimestamp(result.timestamp);
      lastRunTimeElement.textContent = formattedTime;
      lastRunLink.setAttribute("href", result.url)

    } else {
      console.log('No completed runs found');
    }
  });
</script>
