import React, { useState, useEffect } from 'react';
import { Select, Spin, Empty } from 'antd';
import { debounce } from 'lodash';
import axios from 'axios';

const { Option } = Select;

interface AssetSelectorProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  maxCount?: number;
  placeholder?: string;
}

interface AssetOption {
  code: string;
  name: string;
  type: string;
}

const AssetSelector: React.FC<AssetSelectorProps> = ({
  value = [],
  onChange,
  maxCount = 6,
  placeholder = '请选择资产（最多6个）'
}) => {
  const [options, setOptions] = useState<AssetOption[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchText, setSearchText] = useState<string>('');

  // 加载所有资产列表
  useEffect(() => {
    const fetchAllAssets = async () => {
      try {
        const response = await axios.get('http://localhost:8000/get_all_assets/');
        if (response.data && response.data.assets) {
          setOptions(response.data.assets);
        }
      } catch (error) {
        console.error('获取资产列表失败:', error);
      }
    };

    fetchAllAssets();
  }, []);

  // 搜索资产的防抖函数
  const debouncedSearch = debounce(async (searchValue: string) => {
    if (!searchValue) return;

    setLoading(true);
    try {
      const response = await axios.get(`http://localhost:8000/search_top_assets/${searchValue}`);
      if (response.data && response.data.assets) {
        setOptions(response.data.assets);
      }
    } catch (error) {
      console.error('搜索资产失败:', error);
    } finally {
      setLoading(false);
    }
  }, 500);

  // 处理搜索输入
  const handleSearch = (value: string) => {
    setSearchText(value);
    if (value) {
      debouncedSearch(value);
    } else {
      // 如果搜索框被清空，恢复所有资产列表
      axios.get('http://localhost:8000/get_all_assets/')
        .then(response => {
          if (response.data && response.data.assets) {
            setOptions(response.data.assets);
          }
        })
        .catch(error => {
          console.error('获取资产列表失败:', error);
        });
    }
  };

  // 处理选择变化
  const handleChange = (selectedValues: string[]) => {
    if (selectedValues.length > maxCount) {
      selectedValues = selectedValues.slice(0, maxCount);
    }
    onChange && onChange(selectedValues);
  };

  // 过滤选项，防止重复选择
  const filteredOptions = options.filter(
    option => !value.includes(option.code)
  );

  return (
    <Select
      mode="multiple"
      value={value}
      placeholder={placeholder}
      onChange={handleChange}
      onSearch={handleSearch}
      style={{ width: '100%' }}
      filterOption={false}
      notFoundContent={loading ? <Spin size="small" /> : <Empty description="没有找到相关资产" />}
      maxTagCount={6}
      showSearch
      allowClear
    >
      {filteredOptions.map(option => (
        <Option key={option.code} value={option.code}>
          {`${option.name} (${option.code}) [${option.type}]`}
        </Option>
      ))}
    </Select>
  );
};

export default AssetSelector; 