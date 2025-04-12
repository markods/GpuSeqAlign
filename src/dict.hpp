#ifndef DICT_HPP
#define DICT_HPP

#include <functional>
#include <iterator>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

template <
    typename K,
    typename V,
    typename Hash = std::hash<K>,
    typename KeyEqual = std::equal_to<K>,
    typename Allocator = std::allocator<std::pair<const K, V>>>
class Dict
{
private:
    using Entry = std::pair<K, V>;
    using ListAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<Entry>;
    using ListType = std::list<Entry, ListAllocator>;
    using Iterator = typename ListType::iterator;

    struct MapValue
    {
        Iterator iter;
    };

    using MapAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<const K, MapValue>>;
    using MapType = std::unordered_map<K, MapValue, Hash, KeyEqual, MapAllocator>;

    ListType entries;
    MapType index;

public:
    Dict() = default;

    Dict(const Dict& other)
        : entries(other.entries), index()
    {
        rebuild_index();
    }
    Dict(Dict&& other) noexcept
        : entries(std::move(other.entries)), index(std::move(other.index))
    { }

    Dict& operator=(const Dict& other)
    {
        if (this != &other)
        {
            entries = other.entries;
            rebuild_index();
        }
        return *this;
    }
    Dict& operator=(Dict&& other) noexcept
    {
        if (this != &other)
        {
            entries = std::move(other.entries);
            index = std::move(other.index);
        }
        return *this;
    }

    Dict(std::initializer_list<std::pair<const K, V>> init)
    {
        for (const auto& pair : init)
        {
            insert(pair.first, pair.second);
        }
    }

    void swap(Dict& other) noexcept
    {
        entries.swap(other.entries);
        index.swap(other.index);
    }

    void insert(const K& key, const V& value)
    {
        auto it = index.find(key);
        if (it != index.end())
        {
            it->second.iter->second = value;
        }
        else
        {
            entries.emplace_back(key, value);
            Iterator iter = std::prev(entries.end());
            index.emplace(key, MapValue {iter});
        }
    }

    void insert(K&& key, V&& value)
    {
        auto it = index.find(key);
        if (it != index.end())
        {
            it->second.iter->second = std::move(value);
        }
        else
        {
            entries.emplace_back(std::move(key), std::move(value));
            Iterator iter = std::prev(entries.end());
            index.emplace(iter->first, MapValue {iter});
        }
    }

    V& at(const K& key)
    {
        auto it = index.find(key);
        if (it == index.end())
        {
            throw std::out_of_range("Dict::at: key not found");
        }
        return it->second.iter->second;
    }

    const V& at(const K& key) const
    {
        auto it = index.find(key);
        if (it == index.end())
        {
            throw std::out_of_range("Dict::at: key not found");
        }
        return it->second.iter->second;
    }

    V& operator[](const K& key)
    {
        auto it = index.find(key);
        if (it == index.end())
        {
            insert(key, V {});
        }
        return index[key].iter->second;
    }

    bool contains(const K& key) const
    {
        return index.find(key) != index.end();
    }

    void erase(const K& key)
    {
        auto it = index.find(key);
        if (it != index.end())
        {
            entries.erase(it->second.iter);
            index.erase(it);
        }
    }

    void clear()
    {
        entries.clear();
        index.clear();
    }

    std::vector<K> keys() const
    {
        std::vector<K> result {};
        result.reserve(entries.size());

        for (const auto& entry : entries)
        {
            result.push_back(entry.first);
        }

        return result;
    }

    std::vector<V> values() const
    {
        std::vector<V> result {};
        result.reserve(entries.size());

        for (const auto& entry : entries)
        {
            result.push_back(entry.second);
        }

        return result;
    }

    size_t size()
    {
        return entries.size();
    }

    auto begin() -> typename ListType::iterator
    {
        return entries.begin();
    }
    auto end() -> typename ListType::iterator
    {
        return entries.end();
    }
    auto begin() const -> typename ListType::const_iterator
    {
        return entries.begin();
    }
    auto end() const -> typename ListType::const_iterator
    {
        return entries.end();
    }
    auto cbegin() const -> typename ListType::const_iterator
    {
        return entries.cbegin();
    }
    auto cend() const -> typename ListType::const_iterator
    {
        return entries.cend();
    }

    auto rbegin() -> typename ListType::reverse_iterator
    {
        return entries.rbegin();
    }
    auto rend() -> typename ListType::reverse_iterator
    {
        return entries.rend();
    }
    auto rbegin() const -> typename ListType::const_reverse_iterator
    {
        return entries.rbegin();
    }
    auto rend() const -> typename ListType::const_reverse_iterator
    {
        return entries.rend();
    }
    auto crbegin() const -> typename ListType::const_reverse_iterator
    {
        return entries.crbegin();
    }
    auto crend() const -> typename ListType::const_reverse_iterator
    {
        return entries.crend();
    }

    auto find(const K& key) -> typename ListType::iterator
    {
        auto it = index.find(key);
        return (it != index.end()) ? it->second.iter : entries.end();
    }
    auto find(const K& key) const -> typename ListType::const_iterator
    {
        auto it = index.find(key);
        return (it != index.end()) ? it->second.iter : entries.end();
    }

private:
    void rebuild_index()
    {
        index.clear();
        for (auto it = entries.begin(); it != entries.end(); ++it)
        {
            index.emplace(it->first, MapValue {it});
        }
    }
};

template <typename K, typename V, typename H, typename E, typename A>
void swap(Dict<K, V, H, E, A>& lhs, Dict<K, V, H, E, A>& rhs) noexcept
{
    lhs.swap(rhs);
}

#endif // DICT_HPP
